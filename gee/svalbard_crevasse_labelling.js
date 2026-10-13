


// Change this to the digitizing year to get a Sentinel-2 mosiac for it
var year = 2025;
// Run the export tasks
var run_export = true;
/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 */
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

// Map the function over a month of data and take the median.
// Load Sentinel-2 TOA reflectance data (adjusted for processing changes
// that occurred after 2022-01-25).
var dataset = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                  .filterDate(year + '-07-01', year + '-08-31')
                  .filterBounds(ee.Geometry.BBox(10, 76, 30, 81))
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds);

var rgbVis = {
  min: -2,
  max: 0,
  bands: ['B4', 'B3', 'B2'],
};

//Map.setCenter(15, 78, 5);
Map.addLayer(dataset.mosaic().log10(), rgbVis, 'RGB');


if (run_export) {
  // ============================================================
  // CREVASSE TRAINING LABEL EXPORT
  //
  // Merges every poly_pos_*/poly_neg_* Import in this script into
  // one tagged FeatureCollection and exports it as:
  //
  //   1. a GEE asset, for svalbard_crevasse_detection.js to train on;
  //   2. a Drive GeoJSON, for offline spatial/temporal coverage auditing
  //      (see src/label_audit.py in the main repo).
  //
  // Positive polygons are additionally tagged as large crevasses when
  // they intersect a point in the corresponding big_crevasses_<year>
  // FeatureCollection. Negative polygons are always assigned
  // big_crevasses = false and do not undergo a spatial intersection test.
  //
  // Run this from whichever GEE script currently holds the poly_pos_*,
  // poly_neg_*, and big_crevasses_* Imports. Edit LABEL_YEARS below to
  // match exactly the years present as Imports here. An entry referencing
  // a variable that does not exist will prevent the script from loading.
  // ============================================================

  var LABELS_ASSET_ID =
    'users/erikmann/crevasse_training_labels';

  var DRIVE_FOLDER = 'crevasse_maps';
  var DRIVE_FILENAME = 'crevasse_training_labels';

  // Add or remove entries to match the Imports present in this script.
  //
  // year is a plain client-side number rather than an ee.Number, allowing
  // it to be stored directly as a numeric server-side property.
  //
  // big is the point FeatureCollection identifying especially large
  // positive crevasse polygons for that year.
  var LABEL_YEARS = [
    {
      year: 2017,
      neg: poly_neg_2017,
      pos: poly_pos_2017,
      big: big_crevasses_2017
    },
    {
      year: 2019,
      neg: poly_neg_2019,
      pos: poly_pos_2019,
      big: big_crevasses_2019
    },
    {
      year: 2021,
      neg: poly_neg_2021,
      pos: poly_pos_2021,
      big: big_crevasses_2021
    },
    {
      year: 2023,
      neg: poly_neg_2023,
      pos: poly_pos_2023,
      big: big_crevasses_2023
    },
    {
      year: 2024,
      neg: poly_neg_2024,
      pos: poly_pos_2024,
      big: big_crevasses_2024
    },
    {
      year: 2025,
      neg: poly_neg_2025,
      pos: poly_pos_2025,
      big: big_crevasses_2025
    }
  ];

  /**
   * Tags a polygon FeatureCollection for export.
   *
   * Every output feature receives:
   *
   *   year:           digitizing year;
   *   crevassed:      true for positive polygons, false for negatives;
   *   big_crevasses:  true if a positive polygon contains or touches at
   *                    least one marked big-crevasse point;
   *   polyg_id:       unique identifier based on year, class, and the
   *                    original system:index.
   *
   * The source feature is rebuilt from its geometry so that digitizing-tool
   * bookkeeping properties are not included in the export. system:index is
   * retained indirectly as part of polyg_id.
   *
   * For positive polygons, all big-crevasse points from the corresponding
   * year are combined into a single geometry. Each polygon is then tested
   * once against that geometry. Negative polygons skip the spatial test.
   *
   * @param {ee.FeatureCollection} regions Polygon collection to tag.
   * @param {number} year Digitizing year.
   * @param {boolean} crevassed Whether these are positive polygons.
   * @param {ee.FeatureCollection|null} bigCrevassePoints Points marking
   *     especially large positive polygons; unused for negatives.
   * @return {ee.FeatureCollection} Tagged polygon collection.
   */
  function taggedCollection(
    regions,
    year,
    crevassed,
    bigCrevassePoints
  ) {
    var classLabel = crevassed ? 'pos' : 'neg';

    // Build this prefix client-side. Only system:index is server-side and
    // therefore needs to be converted with ee.String() inside map().
    var idPrefix = year + '_' + classLabel + '_';

    var polygons = ee.FeatureCollection(regions);

    // Negative polygons cannot represent large crevasses, so assign false
    // directly and avoid running an unnecessary spatial intersection.
    if (!crevassed) {
      return polygons.map(function(f) {
        f = ee.Feature(f);

        var polygId = ee.String(idPrefix)
          .cat(ee.String(f.get('system:index')));

        return ee.Feature(f.geometry()).set({
          year: year,
          crevassed: false,
          big_crevasses: false,
          polyg_id: polygId
        });
      });
    }

    // Combining the small yearly point collection into one geometry allows
    // each positive polygon to be tested in a single intersects() operation.
    var bigPointsGeometry = ee.FeatureCollection(bigCrevassePoints)
      .geometry();

    return polygons.map(function(f) {
      f = ee.Feature(f);

      var polygId = ee.String(idPrefix)
        .cat(ee.String(f.get('system:index')));

      // intersects() also returns true for a point lying exactly on a
      // polygon boundary.
      var isBigCrevasse = f.geometry().intersects({
        right: bigPointsGeometry,
        maxError: 1
      });

      return ee.Feature(f.geometry()).set({
        year: year,
        crevassed: true,
        big_crevasses: isBigCrevasse,
        polyg_id: polygId
      });
    });
  }

  // Merge every year's negative and positive polygon collections.
  //
  // Negatives are assigned big_crevasses = false directly. Positives are
  // checked against the big-crevasse points from the corresponding year.
  var allLabels = LABEL_YEARS.reduce(function(merged, entry) {
    var negativeLabels = taggedCollection(
      entry.neg,
      entry.year,
      false,
      null
    );

    var positiveLabels = taggedCollection(
      entry.pos,
      entry.year,
      true,
      entry.big
    );

    return merged
      .merge(negativeLabels)
      .merge(positiveLabels);
  }, ee.FeatureCollection([]));

  // Keep console evaluation deliberately minimal. Printing entire features,
  // repeatedly filtering the completed collection, or adding diagnostic
  // layers can make the Code Editor appear unresponsive while Earth Engine
  // evaluates the requested results.
  print('Total tagged labels', allLabels.size());

  Export.table.toAsset({
    collection: allLabels,
    description: 'crevasse_training_labels_to_asset',
    assetId: LABELS_ASSET_ID,
    overwrite: true
  });

  Export.table.toDrive({
    collection: allLabels,
    description: 'crevasse_training_labels_to_drive',
    folder: DRIVE_FOLDER,
    fileNamePrefix: DRIVE_FILENAME,
    fileFormat: 'GeoJSON'
  });
}

