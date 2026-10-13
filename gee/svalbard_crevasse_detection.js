



// ============================================================
// CREVASSE DETECTION RF ENSEMBLE
// Modes:
//   prototype  - tiny in-memory run with map preview
//   training   - train full ensemble, export tree table and validation
//   prediction - load saved tree table, export rasters
// ============================================================


// ============================================================
// TOP-LEVEL RUN SETTINGS
// ============================================================

var RUN_MODE = 'prediction';   // 'prototype', 'training', or 'prediction'

// Per-model console output is expensive. Every print() of a server-side
// object forces Earth Engine to evaluate the whole chain behind it --
// sampling, training, classification -- in the Code Editor, separately
// from and in addition to the export that needs the same result. At 30
// models that is ~300 extra evaluations per training run, and it is
// billed to interactive usage rather than appearing in the Tasks list,
// so it is easy to miss when reconciling quota.
//
// The worst offenders are exportModelTable()'s size()/first() calls,
// which train the entire ensemble just to report a count, and
// runValidation()/runHoldoutValidation()'s per-model confusion matrices.
//
// Leave off for production runs. Turn on to debug a specific model.
var VERBOSE = false;

// print() only when VERBOSE. Takes the same arguments as print().
function verbosePrint() {
  if (VERBOSE) {
    print.apply(null, arguments);
  }
}

// ============================================================
// PREDICTOR EXPERIMENT
// ============================================================

// 'alphaearth' reproduces the current workflow.
// 'sentinel1' uses within-year seasonal Sentinel-1 features.
var INPUT_SOURCE = 'alphaearth';

// AlphaEarth keeps the existing export names. Sentinel-1 exports receive
// a clear suffix so model tables, validation CSVs, and prediction rasters
// cannot be confused with the embedding-based outputs.
var SOURCE_SUFFIX = INPUT_SOURCE === 'sentinel1' ? '_sentinel1' : '';

print('INPUT_SOURCE', INPUT_SOURCE);
print('SOURCE_SUFFIX', SOURCE_SUFFIX);

// Manually redefine this for prediction exports.
// Examples:
//var YEARS = [2021];
//   var YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

var MODEL_TABLE_ASSET_ID =
  'users/erikmann/crevasse_rf_ensemble_all_trees' + SOURCE_SUFFIX;


// ============================================================
// MODE CONFIGURATION
// ============================================================

var CONFIGS = {
  prototype: {
    nModels: 3,
    targetPerClassYearTrain: 100,
    targetPerClassYearTest: 30,

    trainModels: true,
    loadModels: false,
    runValidation: true,
    exportModelTable: false,
    exportValidation: false,
    runHoldoutValidation: false,
    exportHoldoutValidation: false,
    exportDebugSamples: false,
    exportRasters: false
  },

  training: {
    nModels: 30,
    targetPerClassYearTrain: 3000,
    targetPerClassYearTest: 1500,

    trainModels: true,
    loadModels: false,
    runValidation: true,
    exportModelTable: true,
    exportValidation: true,
    runHoldoutValidation: true,
    exportHoldoutValidation: true,
    // Independent of the trained models -- costs nothing extra alongside
    // this run. See buildDebugSamples()/exportDebugSamples().
    exportDebugSamples: true,
    exportRasters: false
  },

  prediction: {
    nModels: 30,
    targetPerClassYearTrain: null,
    targetPerClassYearTest: 1000,

    trainModels: false,
    loadModels: true,
    runValidation: false,
    exportModelTable: false,
    exportValidation: false,
    // loadModelsFromTreeTable() already provides classClf, so holdout
    // validation can be regenerated here without retraining.
    runHoldoutValidation: false,
    exportHoldoutValidation: false,
    exportDebugSamples: false,
    exportRasters: true
  }
};

var CONFIG = CONFIGS[RUN_MODE];

print('RUN_MODE', RUN_MODE);
print('CONFIG', CONFIG);
print('YEARS', YEARS);


// ============================================================
// GLOBAL CONSTANTS
// ============================================================

var SCALE = 10;
var EXPORT_SCALE = 50;
var ONLY_SVALBARD = true;

var COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL';

var EXPORT_CRS = 'EPSG:32633';

var SVALBARD_BOUNDS = ee.Geometry.Rectangle(
  [390000, 8480000, 850000, 8970000],
  EXPORT_CRS,
  false
);
var SVALBARD_EXPORT_CRS_TRANSFORM = [
  EXPORT_SCALE, 0, 390000,
  0, -EXPORT_SCALE, 8970000
];

// Polygon holdout split.
var TRAIN_POLY_FRACTION = 0.7;
var SPLIT_SEED = 12345;

// Capped linear area weighting. This is to avoid over-/undersampling of polygons of various size.
var AREA_MIN_KM2 = 0.35;
var AREA_CAP_KM2 = 100.0;
var WEIGHT_MIN = 8;
var WEIGHT_MAX = 120;

// Random forest parameters.
var RF_TREES = 50;
var RF_BAG_FRACTION = 0.6;
var RF_MIN_LEAF_POPULATION = 4;
var RF_VARIABLES_PER_SPLIT = null;

// Sampling memory helper.
var TILE_SCALE = 4;

// ============================================================
// INPUT TRAINING POLYGONS
// Loaded from the merged label asset (see
// scripts/svalbard_crevasse_labels_export.js), not individual Imports.
// ============================================================

var LABELS_ASSET_ID = 'users/erikmann/crevasse_training_labels';
var TRAINING_YEARS = [2017, 2019, 2021, 2023, 2025]; // 2024 is the intentional holdout
var HOLDOUT_YEARS = [2024]; // Labeled but never trained on -- temporal generalization check.

var allLabels = ee.FeatureCollection(LABELS_ASSET_ID);

var TRAIN_YEARS = TRAINING_YEARS.map(function(year) {
  var yearLabels = allLabels.filter(ee.Filter.eq('year', year));
  return {
    year: year,
    neg: yearLabels.filter(ee.Filter.eq('crevassed', false)),
    pos: yearLabels.filter(ee.Filter.eq('crevassed', true))
  };
});

var n_positive_polygons = ee.Number(0);
var n_negative_polygons = ee.Number(0);
for (var i in TRAIN_YEARS) {
  n_negative_polygons = n_negative_polygons.add(TRAIN_YEARS[i].neg.filterBounds(SVALBARD_BOUNDS).size());
  n_positive_polygons = n_positive_polygons.add(TRAIN_YEARS[i].pos.filterBounds(SVALBARD_BOUNDS).size());
}
print("N negative polygons: ", n_negative_polygons, "N positive polygons: ", n_positive_polygons);


// ============================================================
// BASIC HELPERS
// ============================================================

function getYearImage(year) {
  
  if (INPUT_SOURCE == "sentinel1") {
    var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(SVALBARD_BOUNDS)
      .filterDate(year + "-01-01", year + "-12-31")
      .filter(
        ee.Filter.and(
          ee.Filter.eq("instrumentMode", "IW"),
          ee.Filter.listContains("transmitterReceiverPolarisation", "HV"),
          ee.Filter.eq("orbitProperties_pass", "ASCENDING")
        )
      )
    
    var spring = s1.filterDate(year + "-01-01", year + "-05-31");;
    var summer = s1.filterDate(year + "-06-01", year + "-09-30");
    var fall = s1.filterDate(year + "-10-01", year + "-12-31");
    
    var img = spring.select(["HH", "HV"]).median().rename(["HH_spring", "HV_spring"])
      .addBands(summer.select(["HH", "HV"]).median().rename(["HH_summer", "HV_summer"]))
      .addBands(fall.select(["HH", "HV"]).median().rename(["HH_fall", "HV_fall"]));
    return img;
    
  } else {
    return ee.ImageCollection(COLLECTION_ID)
      .filterDate(
        ee.Date.fromYMD(year, 1, 1),
        ee.Date.fromYMD(year + 1, 1, 1)
      )
      .mosaic();
  }
}

function cappedLinearWeight(areaKm2) {
  areaKm2 = ee.Number(areaKm2);

  var aEff = areaKm2
    .min(AREA_CAP_KM2)
    .max(AREA_MIN_KM2);

  var frac = aEff
    .subtract(AREA_MIN_KM2)
    .divide(ee.Number(AREA_CAP_KM2).subtract(AREA_MIN_KM2));

  return ee.Number(WEIGHT_MIN).add(
    frac.multiply(ee.Number(WEIGHT_MAX).subtract(WEIGHT_MIN))
  );
}

function safeDivide(num, den) {
  num = ee.Number(num);
  den = ee.Number(den);

  return ee.Algorithms.If(
    den.neq(0),
    num.divide(den),
    null
  );
}


// ============================================================
// POLYGON PREPARATION AND SAMPLING
// ============================================================

function preparePolygons(regions, label, year, className, splitSeed) {
  var fc = ee.FeatureCollection(regions);

  if (ONLY_SVALBARD) {
    fc = fc.filterBounds(SVALBARD_BOUNDS);
  }

  fc = fc
    .map(function(f) {
      var areaKm2 = f.geometry().area(1).divide(1e6);

      return ee.Feature(f).set({
        label: label,
        class_name: className,
        year: year,
        area_km2: areaKm2
      });
    })
    .randomColumn(
      'poly_random',
      splitSeed + year * 10 + label
    )
    .map(function(f) {
      var isTrain = ee.Number(f.get('poly_random'))
        .lt(TRAIN_POLY_FRACTION);

      return ee.Feature(f).set({
        poly_split: ee.Algorithms.If(isTrain, 'train', 'test')
      });
    });

  return fc;
}

function buildAllPolygons(splitSeed) {
  var allPolys = ee.FeatureCollection([]);

  for (var i = 0; i < TRAIN_YEARS.length; i++) {
    var y = TRAIN_YEARS[i].year;

    var neg = preparePolygons(TRAIN_YEARS[i].neg, 0, y, 'negative', splitSeed);
    var pos = preparePolygons(TRAIN_YEARS[i].pos, 1, y, 'positive', splitSeed);

    allPolys = allPolys.merge(neg).merge(pos);
  }

  return allPolys;
}

function addSamplingWeights(polygons, targetTotal) {
  polygons = ee.FeatureCollection(polygons)
    .map(function(f) {
      var w = cappedLinearWeight(f.get('area_km2'));
      return ee.Feature(f).set('sample_weight_raw', w);
    });

  var sumW = ee.Number(polygons.aggregate_sum('sample_weight_raw'));

  polygons = polygons.map(function(f) {
    var w = ee.Number(f.get('sample_weight_raw'));

    var n = ee.Number(targetTotal)
      .multiply(w)
      .divide(sumW)
      .round()
      .max(1)
      .int();

    return ee.Feature(f).set('n_samples', n);
  });

  return polygons;
}

function samplePolygons(img, polygons, seed, targetTotal, includeGeometry) {
  includeGeometry = includeGeometry || false;
  polygons = ee.FeatureCollection(polygons);
  var nPoly = polygons.size();

  return ee.FeatureCollection(
    ee.Algorithms.If(
      nPoly.gt(0),
      sampleNonEmptyPolygons(img, polygons, seed, targetTotal, includeGeometry),
      ee.FeatureCollection([])
    )
  );
}

function sampleNonEmptyPolygons(img, polygons, seed, targetTotal, includeGeometry) {
  includeGeometry = includeGeometry || false;
  polygons = addSamplingWeights(polygons, targetTotal);

  var nPoly = polygons.size();
  var polyList = polygons.toList(nPoly);

  var sampled = ee.FeatureCollection(
    ee.List.sequence(0, nPoly.subtract(1)).iterate(
      function(i, acc) {
        i = ee.Number(i);
        acc = ee.FeatureCollection(acc);

        var poly = ee.Feature(polyList.get(i));
        var n = ee.Number(poly.get('n_samples')).int();

        var polySeed = ee.Number(seed)
          .add(i.multiply(1009))
          .int();

        var pts = img.sample({
          region: poly.geometry(),
          scale: SCALE,
          numPixels: n,
          seed: polySeed,
          tileScale: TILE_SCALE,
          geometries: includeGeometry
        }).map(function(p) {
          return ee.Feature(p).set({
            label: poly.get('label'),
            class_name: poly.get('class_name'),
            year: poly.get('year'),
            area_km2: poly.get('area_km2'),
            polyg_id: poly.get('polyg_id'),
            n_samples_requested: n,
            poly_random: poly.get('poly_random'),
            poly_split: poly.get('poly_split'),
            // Free: already a property on the label asset. Lets the
            // debug export be stratified by crevasse size class offline.
            big_crevasses: poly.get('big_crevasses'),
            source_poly_id: poly.id()
          });
        });

        return acc.merge(pts);
      },
      ee.FeatureCollection([])
    )
  );

  return sampled;
}

function buildSampleSet(allPolys, splitName, modelSeed, targetPerClassYear) {
  var samples = ee.FeatureCollection([]);

  for (var i = 0; i < TRAIN_YEARS.length; i++) {
    var y = TRAIN_YEARS[i].year;
    var img = getYearImage(y);

    for (var label = 0; label <= 1; label++) {
      var groupPolys = allPolys
        .filter(ee.Filter.eq('year', y))
        .filter(ee.Filter.eq('label', label))
        .filter(ee.Filter.eq('poly_split', splitName));

      var groupSeed = modelSeed + y * 100 + label * 10000;

      var groupSamples = samplePolygons(
        img,
        groupPolys,
        groupSeed,
        targetPerClassYear
      );

      samples = samples.merge(groupSamples);
    }
  }

  return samples;
}


// ============================================================
// TRAINING
// ============================================================

function trainOneModel(modelIndex, allPolys, bands) {
  var modelSeed = 1000 + modelIndex * 100;

  var trainSamples = buildSampleSet(
    allPolys,
    'train',
    modelSeed,
    CONFIG.targetPerClassYearTrain
  );

  verbosePrint('Model ' + modelIndex + ' training sample count', trainSamples.size());

  var classClf = ee.Classifier.smileRandomForest({
    numberOfTrees: RF_TREES,
    variablesPerSplit: RF_VARIABLES_PER_SPLIT,
    minLeafPopulation: RF_MIN_LEAF_POPULATION,
    bagFraction: RF_BAG_FRACTION,
    seed: modelSeed
  }).train({
    features: trainSamples,
    classProperty: 'label',
    inputProperties: bands
  });

  // Use individual tree votes for prediction. This gives freshly trained
  // models the same prediction path as models reconstructed from saved trees.
  var rawClf = ee.Classifier(
    classClf.setOutputMode('RAW')
  );

  return {
    classClf: classClf,
    rawClf: rawClf
  };
}

function trainModels(bands) {
  var models = [];

  for (var m = 0; m < CONFIG.nModels; m++) {
    var splitSeed = SPLIT_SEED + m * 1000;
    var allPolys = buildAllPolygons(splitSeed);

    verbosePrint('Model ' + m + ' split seed', splitSeed);
    verbosePrint(
      'Model ' + m + ' training polygons',
      allPolys.filter(ee.Filter.eq('poly_split', 'train')).size()
    );
    verbosePrint(
      'Model ' + m + ' testing polygons',
      allPolys.filter(ee.Filter.eq('poly_split', 'test')).size()
    );

    models.push({
      modelIndex: m,
      splitSeed: splitSeed,
      allPolys: allPolys,
      modelObj: trainOneModel(m, allPolys, bands)
    });
  }

  return models;
}


// ============================================================
// MODEL TABLE EXPORT AND LOADING
// ============================================================

function modelTreesToFeatureCollection(modelRecord) {
  var modelId = modelRecord.modelIndex;

  var trees = ee.List(
    ee.Dictionary(modelRecord.modelObj.classClf.explain()).get('trees')
  );

  var treeIds = ee.List.sequence(0, trees.size().subtract(1));

  return ee.FeatureCollection(
    treeIds.map(function(treeId) {
      treeId = ee.Number(treeId);

      // The fake point geometry is needed to export this as a table.
      return ee.Feature(ee.Geometry.Point([0, 0]), {
        model_id: modelId,
        tree_id: treeId,
        tree: trees.get(treeId),

        input_source: INPUT_SOURCE,

        n_models: CONFIG.nModels,
        rf_number_of_trees: RF_TREES,
        rf_bag_fraction: RF_BAG_FRACTION,
        rf_variables_per_split: RF_VARIABLES_PER_SPLIT,
        rf_min_leaf_population: RF_MIN_LEAF_POPULATION,

        split_seed: modelRecord.splitSeed,
        train_poly_fraction: TRAIN_POLY_FRACTION,
        target_per_class_year_train: CONFIG.targetPerClassYearTrain,

        area_min_km2: AREA_MIN_KM2,
        area_cap_km2: AREA_CAP_KM2,
        weight_min: WEIGHT_MIN,
        weight_max: WEIGHT_MAX
      });
    })
  );
}

function exportModelTable(models) {
  var allTreeFeatures = ee.FeatureCollection([]);

  for (var m = 0; m < CONFIG.nModels; m++) {
    allTreeFeatures = allTreeFeatures.merge(
      modelTreesToFeatureCollection(models[m])
    );
  }

  verbosePrint('All exported RF trees', allTreeFeatures.size());
  verbosePrint('Example exported tree feature', allTreeFeatures.first());

  Export.table.toAsset({
    collection: allTreeFeatures,
    description: 'crevasse_rf_ensemble_all_trees' + SOURCE_SUFFIX,
    assetId: MODEL_TABLE_ASSET_ID,
    overwrite: true
  });
}

function loadOneModelFromTreeTable(treeTable, modelId) {
  var treeStrings = treeTable
    .filter(ee.Filter.eq('model_id', modelId))
    .sort('tree_id')
    .aggregate_array('tree');

  var classClf = ee.Classifier.decisionTreeEnsemble(treeStrings);

  var rawClf = ee.Classifier(
    classClf.setOutputMode('RAW')
  );

  return {
    modelIndex: modelId,
    modelObj: {
      classClf: classClf,
      rawClf: rawClf
    }
  };
}

function loadModelsFromTreeTable() {
  var treeTable = ee.FeatureCollection(MODEL_TABLE_ASSET_ID);

  verbosePrint('Loaded model tree table', treeTable.limit(5));
  verbosePrint('Loaded model tree count', treeTable.size());

  var models = [];

  for (var m = 0; m < CONFIG.nModels; m++) {
    models.push(loadOneModelFromTreeTable(treeTable, m));
  }

  return models;
}


// ============================================================
// CLASSIFICATION HELPERS
// ============================================================

function classifyPositiveVoteFraction(img, modelRecord) {
  // RAW returns one class vote per tree. Because the class labels are 0 and 1,
  // the mean of (vote == 1) is the positive-class tree-vote fraction.
  var rawVotes = img
    .classify(modelRecord.modelObj.rawClf)
    .rename('raw_votes');

  return rawVotes
    .eq(1)
    .arrayReduce(ee.Reducer.mean(), [0])
    .arrayGet([0])
    .rename('positive_vote_fraction');
}


// ============================================================
// VALIDATION
// ============================================================

function confusionMatrixToFeature(cm, modelName, modelId, extraProps) {
  var arr = ee.Array(cm.array());

  var tn = ee.Number(arr.get([0, 0]));
  var fp = ee.Number(arr.get([0, 1]));
  var fn = ee.Number(arr.get([1, 0]));
  var tp = ee.Number(arr.get([1, 1]));

  var total = tn.add(fp).add(fn).add(tp);

  var accuracy = safeDivide(tp.add(tn), total);
  var precision = safeDivide(tp, tp.add(fp));
  var recall = safeDivide(tp, tp.add(fn));
  var specificity = safeDivide(tn, tn.add(fp));
  var falsePositiveRate = safeDivide(fp, fp.add(tn));
  var falseNegativeRate = safeDivide(fn, fn.add(tp));

  var f1 = ee.Algorithms.If(
    ee.Number(precision).add(ee.Number(recall)).neq(0),
    ee.Number(2)
      .multiply(ee.Number(precision))
      .multiply(ee.Number(recall))
      .divide(ee.Number(precision).add(ee.Number(recall))),
    null
  );

  var baseProps = {
    model_name: modelName,
    model_id: modelId,

    tn: tn,
    fp: fp,
    fn: fn,
    tp: tp,
    n_test: total,

    accuracy: accuracy,
    precision_crevasse: precision,
    recall_crevasse: recall,
    specificity_noncrevasse: specificity,
    false_positive_rate: falsePositiveRate,
    false_negative_rate: falseNegativeRate,
    f1_crevasse: f1
  };

  return ee.Feature(
    null,
    ee.Dictionary(baseProps).combine(extraProps, true)
  );
}

function runValidation(models) {
  var validationRows = ee.FeatureCollection([]);

  for (var vm = 0; vm < CONFIG.nModels; vm++) {
    var testSamples = buildSampleSet(
      models[vm].allPolys,
      'test',
      999999 + vm * 1000,
      CONFIG.targetPerClassYearTest
    );

    var evalSingle = testSamples.classify(
      models[vm].modelObj.classClf,
      'classification'
    );

    var cmSingle = evalSingle.errorMatrix(
      'label',
      'classification',
      [0, 1]
    );

    var row = confusionMatrixToFeature(
      cmSingle,
      'model_' + vm,
      vm,
      {
        classifier_type: 'smileRandomForest',
        input_source: INPUT_SOURCE,

        validation_type: 'repeated_polygon_holdout',
        split_seed: models[vm].splitSeed,
        n_models: CONFIG.nModels,
        target_per_class_year_test: CONFIG.targetPerClassYearTest
      }
    );

    validationRows = validationRows.merge(ee.FeatureCollection([row]));

    verbosePrint('Model ' + vm + ' repeated-split confusion matrix', cmSingle);
    verbosePrint('Model ' + vm + ' repeated-split accuracy', cmSingle.accuracy());
  }

  verbosePrint('Validation summary table', validationRows);

  return validationRows;
}

function exportValidationRows(validationRows) {
  Export.table.toDrive({
    collection: validationRows,
    description: 'crevasse_rf_ensemble_validation_summary' + SOURCE_SUFFIX,
    folder: 'crevasse_maps',
    fileNamePrefix: 'crevasse_rf_ensemble_validation_summary' + SOURCE_SUFFIX,
    fileFormat: 'CSV'
  });
}

// Temporal holdout validation: evaluate already-trained models against a
// year (e.g. 2024) that was excluded from TRAINING_YEARS entirely, so no
// model has ever seen it. Kept as a separate export from
// exportValidationRows -- it answers a different question (temporal
// generalization to an unseen year) than the repeated_polygon_holdout
// validation above (spatial generalization within trained years), and
// mixing both into one CSV would silently blend two rows per model into
// one misleading average downstream.
function runHoldoutValidation(models, year) {
  var img = getYearImage(year);

  // preparePolygons() also assigns a poly_split field, but it's unused
  // here -- every polygon for this year is already held out, so every one
  // of them is sampled as test data.
  var negPolys = preparePolygons(
    allLabels.filter(ee.Filter.eq('year', year)).filter(ee.Filter.eq('crevassed', false)),
    0,
    year,
    'negative',
    SPLIT_SEED
  );
  var posPolys = preparePolygons(
    allLabels.filter(ee.Filter.eq('year', year)).filter(ee.Filter.eq('crevassed', true)),
    1,
    year,
    'positive',
    SPLIT_SEED
  );

  var holdoutRows = ee.FeatureCollection([]);

  for (var hm = 0; hm < CONFIG.nModels; hm++) {
    var seed = 5000000 + hm * 1000;

    var holdoutSamples = samplePolygons(img, negPolys, seed, CONFIG.targetPerClassYearTest)
      .merge(samplePolygons(img, posPolys, seed + 1, CONFIG.targetPerClassYearTest));

    var evalHoldout = holdoutSamples.classify(
      models[hm].modelObj.classClf,
      'classification'
    );

    var cmHoldout = evalHoldout.errorMatrix(
      'label',
      'classification',
      [0, 1]
    );

    var row = confusionMatrixToFeature(
      cmHoldout,
      'model_' + hm,
      hm,
      {
        classifier_type: 'smileRandomForest',
        input_source: INPUT_SOURCE,

        validation_type: 'temporal_holdout_' + year,
        holdout_year: year,
        n_models: CONFIG.nModels,
        target_per_class_year_test: CONFIG.targetPerClassYearTest
      }
    );

    holdoutRows = holdoutRows.merge(ee.FeatureCollection([row]));

    verbosePrint('Model ' + hm + ' ' + year + ' holdout confusion matrix', cmHoldout);
    verbosePrint('Model ' + hm + ' ' + year + ' holdout accuracy', cmHoldout.accuracy());
  }

  verbosePrint(year + ' holdout validation summary table', holdoutRows);

  return holdoutRows;
}

function exportHoldoutValidationRows(holdoutRows, year) {
  Export.table.toDrive({
    collection: holdoutRows,
    description: 'crevasse_rf_ensemble_validation_holdout_' + year + SOURCE_SUFFIX,
    folder: 'crevasse_maps',
    fileNamePrefix: 'crevasse_rf_ensemble_validation_holdout_' + year + SOURCE_SUFFIX,
    fileFormat: 'CSV'
  });
}


// ============================================================
// DEBUG SAMPLE EXPORT
// A one-time (not per-model) dump of sampled points -- input embedding
// bands plus label/provenance columns -- for offline debugging of
// label/feature mismatches (e.g. checking whether firn-covered crevasse
// labels look like plain firn to the annual embedding). Independent of
// any trained model, so it costs nothing extra to include alongside a
// training or prediction run.
// ============================================================

var DEBUG_TARGET_PER_CLASS_YEAR = 500;
var DEBUG_SEED = 777;

function buildDebugSamples() {
  var debugYears = TRAINING_YEARS.concat(HOLDOUT_YEARS);
  var samples = ee.FeatureCollection([]);

  debugYears.forEach(function(year) {
    var img = getYearImage(year);
    var isHoldout = HOLDOUT_YEARS.indexOf(year) >= 0;

    var negPolys = preparePolygons(
      allLabels.filter(ee.Filter.eq('year', year)).filter(ee.Filter.eq('crevassed', false)),
      0, year, 'negative', SPLIT_SEED
    );
    var posPolys = preparePolygons(
      allLabels.filter(ee.Filter.eq('year', year)).filter(ee.Filter.eq('crevassed', true)),
      1, year, 'positive', SPLIT_SEED
    );

    // Sample from every polygon for the year (ignoring poly_split) --
    // this is for inspection, not model evaluation, so there's no need to
    // hold anything out. Points carry big_crevasses, so the size-class
    // comparison can be redone offline at the native 10 m embedding scale
    // instead of on the 50 m export.
    var yearSamples = samplePolygons(img, negPolys, DEBUG_SEED, DEBUG_TARGET_PER_CLASS_YEAR, true)
      .merge(samplePolygons(img, posPolys, DEBUG_SEED + 1, DEBUG_TARGET_PER_CLASS_YEAR, true))
      .map(function(f) {
        return f.set('is_holdout_year', isHoldout);
      });

    samples = samples.merge(yearSamples);
  });

  return samples;
}

function exportDebugSamples(samples) {
  Export.table.toDrive({
    collection: samples,
    description: 'crevasse_training_debug_samples' + SOURCE_SUFFIX,
    folder: 'crevasse_maps',
    fileNamePrefix: 'crevasse_training_debug_samples' + SOURCE_SUFFIX,
    fileFormat: 'CSV'
  });
}


// ============================================================
// RASTER PREDICTION
// One uint8 band per ensemble model and year.
// Each value is the fraction of trees voting for class 1.
// Decode any band as: positive tree-vote fraction = value / 255.
// ============================================================

function makeModelVoteFractionStack(year, models) {
  var img = getYearImage(year);
  var voteFractionBands = [];

  for (var mi = 0; mi < CONFIG.nModels; mi++) {
    var bandName = 'prob_m' + ('0' + mi).slice(-2);

    var voteFractionU8 = classifyPositiveVoteFraction(
      img,
      models[mi]
    )
      .multiply(255)
      .round()
      .clamp(0, 255)
      .toUint8()
      .rename(bandName);

    voteFractionBands.push(voteFractionU8);
  }

  return ee.Image.cat(voteFractionBands).set({
    encoding: 'uint8_scaled_positive_tree_vote_fraction',
    value_scale: 1 / 255,
    value_offset: 0,
    positive_class: 1,
    trees_per_model: RF_TREES,
    band_order: 'prob_m00_to_prob_m29',
    year: year,
    n_models: CONFIG.nModels,
    input_source: INPUT_SOURCE,

    source_model_table: MODEL_TABLE_ASSET_ID
  });
}

function exportPredictionRaster(year, voteFractionStack) {
  Export.image.toDrive({
    image: voteFractionStack,
    description: 'crevasse_model_probabilities_' + year + SOURCE_SUFFIX,
    folder: 'crevasse_maps',
    fileNamePrefix: 'crevasse_model_probabilities_' + year + SOURCE_SUFFIX,
    skipEmptyTiles: true,
    region: SVALBARD_BOUNDS,
    crsTransform: SVALBARD_EXPORT_CRS_TRANSFORM,
    scale: EXPORT_SCALE,
    crs: EXPORT_CRS,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
}

function runPrediction(models) {
  YEARS.forEach(function(year) {
    print('Preparing per-model positive vote-fraction stack for year', year);

    var voteFractionStack = makeModelVoteFractionStack(year, models);

    if (CONFIG.exportRasters) {
      exportPredictionRaster(year, voteFractionStack);
    }
  });
}


// ============================================================
// MAIN
// ============================================================

var bands = getYearImage(TRAIN_YEARS[0].year).bandNames();
print('Input bands', bands);

var models;

if (CONFIG.trainModels) {
  models = trainModels(bands);
}

if (CONFIG.loadModels) {
  models = loadModelsFromTreeTable();
}

if (CONFIG.runValidation) {
  var validationRows = runValidation(models);

  if (CONFIG.exportValidation) {
    exportValidationRows(validationRows);
  }
}

if (CONFIG.exportModelTable) {
  exportModelTable(models);
}

if (CONFIG.runHoldoutValidation) {
  HOLDOUT_YEARS.forEach(function(year) {
    var holdoutRows = runHoldoutValidation(models, year);

    if (CONFIG.exportHoldoutValidation) {
      exportHoldoutValidationRows(holdoutRows, year);
    }
  });
}

if (CONFIG.exportDebugSamples) {
  var debugSamples = buildDebugSamples();
  exportDebugSamples(debugSamples);
}

if (CONFIG.exportRasters) {
  runPrediction(models);
}




