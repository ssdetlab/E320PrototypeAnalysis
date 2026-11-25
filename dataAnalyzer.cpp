#include <cmath>

#include "RootTrackAnalyzer.hpp"

int main() {
  std::string prefix = "alignment-bkg-dc-filtered-low-chi2";

  RootTrackAnalyzer::Config analyzerCfg;
  analyzerCfg.chipIds = {10, 12, 14, 16, 18};
  // analyzerCfg.chipIds = {10, 12, 16, 18};
  analyzerCfg.clusterTreeName = "data";
  analyzerCfg.inClusterDir =
      "/home/romanurmanov/work/E320/E320Prototype/"
      "E320Prototype_dataInRootFormat/local_alignment_data/processed/sig";
  analyzerCfg.trackTreeName = "fitted-tracks";
  analyzerCfg.inTrackDir =
      "/home/romanurmanov/work/E320/E320Prototype/E320Prototype_analysis/"
      "data/alignment/local_2026_data/production_alignment/"
      "sig/filtered";

  RootTrackAnalyzer analyzer(analyzerCfg);

  RootTrackAnalyzer::Stats::Options statsOpt{};
  statsOpt.chipIds = {10, 12, 14, 16, 18};
  statsOpt.chipIdsMap = {{8, 10}, {6, 12}, {4, 14}, {2, 16}, {0, 18}};

  double chipHalfX = 14.97088;
  double chipHalfY = 6.88128;

  // X count
  statsOpt.xCountBins = 20;
  statsOpt.xCountMin = 0;
  statsOpt.xCountMax = 20;

  // Mag conf hist parameters
  statsOpt.dipoleStrengthBins = 100;
  statsOpt.dipoleStrengthMin = -0.4;
  statsOpt.dipoleStrengthMax = 0.0;

  statsOpt.xCorrectorStrengthBins = 100;
  statsOpt.xCorrectorStrengthMin = -0.1;
  statsOpt.xCorrectorStrengthMax = 0.1;

  // Track multiplicity hist parameters
  statsOpt.nTracksBins = 110;
  statsOpt.nTracksMin = 0;
  statsOpt.nTracksMax = 110;

  // Cluster multiplicity hist parameters
  statsOpt.nClustersBins = 210;
  statsOpt.nClustersMin = 0;
  statsOpt.nClustersMax = 210;

  // Cluster size hist parameters
  statsOpt.clusterSizeBins = 4;
  statsOpt.clusterSizeMin = 0;
  statsOpt.clusterSizeMax = 4;

  // Chi2 hists parameters
  statsOpt.chi2Bins = 400;
  statsOpt.chi2Min = 0;
  statsOpt.chi2Max = 4000;

  // Local track hits hists parameters
  statsOpt.trackHitsLocalBinsX = 1024;
  statsOpt.trackHitsLocalBinsY = 512;
  statsOpt.trackHitsLocalMinX = -chipHalfX;
  statsOpt.trackHitsLocalMaxX = chipHalfX;
  statsOpt.trackHitsLocalMinY = -chipHalfY;
  statsOpt.trackHitsLocalMaxY = chipHalfY;

  // Track hits error hists parameters
  statsOpt.trackHitsErrorBinsX = 1024;
  statsOpt.trackHitsErrorBinsY = 512;
  statsOpt.trackHitsErrorMinX = 3e-3;
  statsOpt.trackHitsErrorMaxX = 6e-3;
  statsOpt.trackHitsErrorMinY = 3e-3;
  statsOpt.trackHitsErrorMaxY = 6e-3;

  // Track hits leave-one-out variance hists parameters
  statsOpt.trackHitsLooErrorBinsX = 1024;
  statsOpt.trackHitsLooErrorBinsY = 512;
  statsOpt.trackHitsLooErrorMinX = 0.003;
  statsOpt.trackHitsLooErrorMaxX = 0.007;
  statsOpt.trackHitsLooErrorMinY = 0.003;
  statsOpt.trackHitsLooErrorMaxY = 0.007;

  // Resiudals hists parameters
  statsOpt.residualsXBins = 300;
  statsOpt.residualsXMin = -0.1;
  statsOpt.residualsXMax = 0.1;

  statsOpt.residualsYBins = 300;
  statsOpt.residualsYMin = -0.1;
  statsOpt.residualsYMax = 0.1;

  // Leave-one-out resiudals hists parameters
  statsOpt.looResidualsXBins = 300;
  statsOpt.looResidualsXMin = -0.03;
  statsOpt.looResidualsXMax = 0.03;

  statsOpt.looResidualsYBins = 300;
  statsOpt.looResidualsYMin = -0.03;
  statsOpt.looResidualsYMax = 0.03;

  // Pulls hists parameters
  statsOpt.pullsXBins = 100;
  statsOpt.pullsXMin = -5;
  statsOpt.pullsXMax = 5;

  statsOpt.pullsYBins = 100;
  statsOpt.pullsYMin = -5;
  statsOpt.pullsYMax = 5;

  // Momentum hists parameters
  statsOpt.momentumXBins = 300;
  statsOpt.momentumXMin = -0.05;
  statsOpt.momentumXMax = 0.05;

  statsOpt.momentumYBins = 100;
  statsOpt.momentumYMin = 0.03;
  statsOpt.momentumYMax = 0.07;

  statsOpt.momentumZBins = 300;
  statsOpt.momentumZMin = 0;
  statsOpt.momentumZMax = 10;

  // Momentum direction hists parameters
  statsOpt.momentumDirXBins = 300;
  statsOpt.momentumDirXMin = -0.1;
  statsOpt.momentumDirXMax = 0.1;

  statsOpt.momentumDirYBins = 1000;
  statsOpt.momentumDirYMin = 0.02;
  statsOpt.momentumDirYMax = 0.045;

  statsOpt.momentumDirZBins = 300;
  statsOpt.momentumDirZMin = 0.9;
  statsOpt.momentumDirZMax = 1;

  statsOpt.momentumPhiBins = 300;
  statsOpt.momentumPhiMin = 0.02;
  statsOpt.momentumPhiMax = 0.05;

  statsOpt.momentumThetaBins = 300;
  statsOpt.momentumThetaMin = 1.55;
  statsOpt.momentumThetaMax = 1.59;

  // Vertex hists parameters
  statsOpt.vertexXBins = 300;
  statsOpt.vertexXMin = -100;
  statsOpt.vertexXMax = 100;

  statsOpt.vertexYBins = 300;
  statsOpt.vertexYMin = -100;
  statsOpt.vertexYMax = 100;

  RootTrackAnalyzer::Options analyzerOpt;
  analyzerOpt.prefix = prefix;
  analyzerOpt.statsOpt = statsOpt;

  analyzerOpt.nBXs = 54500;  // Sig

  // Plot blocks
  analyzerOpt.plotLeaveOneOut = true;
  analyzerOpt.plotUncertainty = true;
  analyzerOpt.plotCorrelations = true;

  // --------------------------------------------------
  // No cuts
  analyzerOpt.minOcc = 0;
  analyzerOpt.maxOcc = 1e9;
  analyzerOpt.requireEpicsParity = false;
  analyzerOpt.epicsParity = 0;
  analyzerOpt.minXCount = 0;
  analyzerOpt.maxXCount = 20;
  analyzerOpt.minChi2Predicted = 0;
  analyzerOpt.maxChi2Predicted = 1e9;
  analyzerOpt.minChi2Filtered = 0;
  analyzerOpt.maxChi2Filtered = 1e9;
  analyzerOpt.minChi2Smoothed = 0;
  analyzerOpt.maxChi2Smoothed = 1e9;
  analyzerOpt.minVertexZ = -1e9;
  analyzerOpt.maxVertexZ = 1e9;
  analyzerOpt.minVertexY = -1e9;
  analyzerOpt.maxVertexY = 1e9;
  analyzerOpt.minMomentumX = 0.0;
  analyzerOpt.maxMomentumX = 10.0;

  // // --------------------------------------------------
  // // High chi2 filtered
  // analyzerOpt.minOcc = 0;
  // analyzerOpt.maxOcc = 1e9;
  // analyzerOpt.requireEpicsParity = false;
  // analyzerOpt.epicsParity = 0;
  // analyzerOpt.minXCount = 5;
  // analyzerOpt.maxXCount = 5;
  // analyzerOpt.minChi2Predicted = 0;
  // analyzerOpt.maxChi2Predicted = 1e9;
  // analyzerOpt.minChi2Filtered = 0;
  // analyzerOpt.maxChi2Filtered = 1e9;
  // analyzerOpt.minChi2Smoothed = 1750;
  // analyzerOpt.maxChi2Smoothed = 3400;
  // analyzerOpt.minVertexZ = -30;
  // analyzerOpt.maxVertexZ = 0;
  // analyzerOpt.minVertexY = -30;
  // analyzerOpt.maxVertexY = 0;
  // analyzerOpt.minMomentumX = 0.0;
  // analyzerOpt.maxMomentumX = 10.0;

  // // High chi2 aligned
  // analyzerOpt.minOcc = 0;
  // analyzerOpt.maxOcc = 1e3;
  // analyzerOpt.requireEpicsParity = false;
  // analyzerOpt.epicsParity = 0;
  // analyzerOpt.minXCount = 10;
  // analyzerOpt.maxXCount = 10;
  // analyzerOpt.minChi2Predicted = 0;
  // analyzerOpt.maxChi2Predicted = 1e9;
  // analyzerOpt.minChi2Filtered = 0;
  // analyzerOpt.maxChi2Filtered = 1e9;
  // analyzerOpt.minChi2Smoothed = 0;
  // analyzerOpt.maxChi2Smoothed = 30;
  // analyzerOpt.minVertexZ = -20;
  // analyzerOpt.maxVertexZ = 0;
  // analyzerOpt.minVertexY = -20;
  // analyzerOpt.maxVertexY = 0;
  // analyzerOpt.minMomentumX = 0;
  // analyzerOpt.maxMomentumX = 10;

  // // --------------------------------------------------
  // // Low chi2 filtered
  // analyzerOpt.minOcc = 0;
  // analyzerOpt.maxOcc = 1e9;
  // analyzerOpt.requireEpicsParity = false;
  // analyzerOpt.epicsParity = 0;
  // analyzerOpt.minXCount = 5;
  // analyzerOpt.maxXCount = 5;
  // analyzerOpt.minChi2Predicted = 0;
  // analyzerOpt.maxChi2Predicted = 1e9;
  // analyzerOpt.minChi2Filtered = 0;
  // analyzerOpt.maxChi2Filtered = 1e9;
  // analyzerOpt.minChi2Smoothed = 750;
  // analyzerOpt.maxChi2Smoothed = 2000;
  // analyzerOpt.minVertexZ = -10;
  // analyzerOpt.maxVertexZ = 10;
  // analyzerOpt.minVertexY = -10;
  // analyzerOpt.maxVertexY = 10;
  // analyzerOpt.minMomentumX = 1.0;
  // analyzerOpt.maxMomentumX = 10.0;

  // // --------------------------------------------------
  // // Low chi2 aligned
  // analyzerOpt.minOcc = 0;
  // analyzerOpt.maxOcc = 1e9;
  // analyzerOpt.requireEpicsParity = false;
  // analyzerOpt.epicsParity = 0;
  // analyzerOpt.minXCount = 10;
  // analyzerOpt.maxXCount = 10;
  // analyzerOpt.minChi2Predicted = 0;
  // analyzerOpt.maxChi2Predicted = 1e9;
  // analyzerOpt.minChi2Filtered = 0;
  // analyzerOpt.maxChi2Filtered = 1e9;
  // analyzerOpt.minChi2Smoothed = 0;
  // analyzerOpt.maxChi2Smoothed = 1e9;
  // analyzerOpt.minVertexZ = -1e9;
  // analyzerOpt.maxVertexZ = 1e9;
  // analyzerOpt.minVertexY = -1e9;
  // analyzerOpt.maxVertexY = 1e9;
  // analyzerOpt.minMomentumX = 0.0;
  // analyzerOpt.maxMomentumX = 10.0;

  analyzer.process(analyzerOpt);

  return 0;
}
