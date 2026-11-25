#include "SimRootTrackAnalyzer.hpp"

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>

#include <TCanvas.h>

#include "Painters.hpp"

namespace {

double normalPdf(double *x, double *par) {
  double xx = x[0];
  double mu = par[0];
  double sigma = par[1];
  return 1.0 / (std::sqrt(2 * M_PI) * sigma) *
         std::exp(-0.5 * (xx - mu) * (xx - mu) / (sigma * sigma));
}

double binomialLikelihood(double p, std::size_t accept, std::size_t reject) {
  return accept * std::log(p) + reject * std::log(1 - p);
}

std::pair<double, double> getBinomialErrors(std::size_t accept,
                                            std::size_t reject,
                                            double stepSize) {
  double p = static_cast<double>(accept) / (accept + reject);

  double lMax = binomialLikelihood(p, accept, reject);

  double lSearch = lMax;
  int k = 0;
  while (binomialLikelihood(p - k * stepSize, accept, reject) - lMax > -0.5) {
    k++;
  }
  double leftErrBound = p - k * stepSize;

  lSearch = lMax;
  k = 0;
  while (binomialLikelihood(p + k * stepSize, accept, reject) - lMax > -0.5) {
    k++;
  }
  double rightErrBound = p + k * stepSize;

  return {p - leftErrBound, rightErrBound - p};
};

}  // namespace

SimRootTrackAnalyzer::SimRootTrackAnalyzer(const Config &config)
    : m_cfg(config) {
  // Collect track paths
  std::cout << "Collecting track data paths from\n"
            << config.inTrackDir << "\n";
  std::vector<std::string> trackPaths;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(config.inTrackDir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".root") {
      continue;
    }
    trackPaths.push_back(entry.path());
  }
  std::sort(trackPaths.begin(), trackPaths.end());

  // Collect cluster paths
  std::cout << "Collecting cluster data paths from\n"
            << config.inClusterDir << "\n";
  std::vector<std::string> clusterPaths;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(config.inClusterDir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".root") {
      continue;
    }
    clusterPaths.push_back(entry.path());
  }
  std::sort(clusterPaths.begin(), clusterPaths.end());

  // Print debug info
  std::cout << "---------------------------------------------\n";
  std::cout << "Cluster paths:\n";
  for (const auto &path : clusterPaths) {
    std::cout << path << "\n";
  }
  std::cout << "---------------------------------------------\n";
  std::cout << "Track paths:\n";
  for (const auto &path : trackPaths) {
    std::cout << path << "\n";
  }
  std::cout << "---------------------------------------------\n";

  if (clusterPaths.empty()) {
    throw std::invalid_argument("Missing cluster file path");
  }
  if (m_cfg.clusterTreeName.empty()) {
    throw std::invalid_argument("Missing cluster tree name");
  }
  if (trackPaths.empty()) {
    throw std::invalid_argument("Missing track file path");
  }
  if (m_cfg.trackTreeName.empty()) {
    throw std::invalid_argument("Missing track tree name");
  }

  if (clusterPaths.size() == 1) {
    m_clusterFile = new TFile(clusterPaths.at(0).c_str());
    m_clusterTree = m_clusterFile->Get<TTree>(m_cfg.clusterTreeName.c_str());
  } else {
    m_clusterChainOwner = new TChain(m_cfg.clusterTreeName.c_str());
    for (const auto &path : clusterPaths) {
      m_clusterChainOwner->Add(path.c_str());
    }
    m_clusterTree = dynamic_cast<TTree *>(m_clusterChainOwner);
  }
  if (trackPaths.size() == 1) {
    m_trackFile = new TFile(trackPaths.at(0).c_str());
    m_trackTree = m_trackFile->Get<TTree>(m_cfg.trackTreeName.c_str());
  } else {
    m_trackChainOwner = new TChain(m_cfg.trackTreeName.c_str());
    for (const auto &path : trackPaths) {
      m_trackChainOwner->Add(path.c_str());
    }
    m_trackTree = dynamic_cast<TTree *>(m_trackChainOwner);
  }

  //------------------------------------------------------------------
  // Cluster tree branches

  // Event ID
  m_clusterTree->SetBranchAddress("eventId", &m_clusterEventId);

  // Surface geometry ID
  m_clusterTree->SetBranchAddress("geoId", &m_geoId);

  //------------------------------------------------------------------
  // Track tree branches

  // Magnet configuration as seen by track
  m_trackTree->SetBranchAddress("quad1Grad", &m_quad1Grad);
  m_trackTree->SetBranchAddress("quad2Grad", &m_quad2Grad);
  m_trackTree->SetBranchAddress("quad3Grad", &m_quad3Grad);
  m_trackTree->SetBranchAddress("xCorrectorStrength", &m_xCorrectorStrength);
  m_trackTree->SetBranchAddress("dipoleStrength", &m_dipoleStrength);

  // True track hits in the surface frame
  m_trackTree->SetBranchAddress("trueTrackHitsLocal", &m_trueTrackHitsLocal);

  // True track hits in the global frame
  m_trackTree->SetBranchAddress("trueTrackHitsGlobal", &m_trueTrackHitsGlobal);

  // True track momentum on the surface in the track frame
  m_trackTree->SetBranchAddress("onSurfaceMomentumTruth",
                                &m_onSurfaceMomentumTruth);

  // Signal/background flag of the track's clusters
  m_trackTree->SetBranchAddress("isSignal", &m_isSignal);

  // Measurement hits in the surface frame
  m_trackTree->SetBranchAddress("trackHitsLocal", &m_trackHitsLocal);

  // Measurement hits in the global frame
  m_trackTree->SetBranchAddress("trackHitsGlobal", &m_trackHitsGlobal);

  // Direction measurements in the track frame
  m_trackTree->SetBranchAddress("onSurfaceTrackDirection",
                                &m_onSurfaceTrackDirection);

  // Covariances of the track hits
  m_trackTree->SetBranchAddress("trackHitCovs", &m_trackHitCovs);

  // Covariances of the track agnles
  m_trackTree->SetBranchAddress("trackAngleCovs", &m_trackAngleCovs);

  // Geometry ids of the track hits
  m_trackTree->SetBranchAddress("geometryIds", &m_geometryIds);

  // KF predicted track hits in the surface frame
  m_trackTree->SetBranchAddress("predictedTrackHitsLocal",
                                &m_predictedTrackHitsLocal);
  m_trackTree->SetBranchAddress("filteredTrackHitsLocal",
                                &m_filteredTrackHitsLocal);
  m_trackTree->SetBranchAddress("smoothedTrackHitsLocal",
                                &m_smoothedTrackHitsLocal);

  m_trackTree->SetBranchAddress("leaveOneOutTrackHitsLocal",
                                &m_leaveOneOutTrackHitsLocal);

  // KF predicted track hits in the global frame
  m_trackTree->SetBranchAddress("predictedTrackHitsGlobal",
                                &m_predictedTrackHitsGlobal);
  m_trackTree->SetBranchAddress("filteredTrackHitsGlobal",
                                &m_filteredTrackHitsGlobal);
  m_trackTree->SetBranchAddress("smoothedTrackHitsGlobal",
                                &m_smoothedTrackHitsGlobal);

  m_trackTree->SetBranchAddress("leaveOneOutTrackHitsGlobal",
                                &m_leaveOneOutTrackHitsGlobal);

  // KF predicted on surface momenta in the track frame
  m_trackTree->SetBranchAddress("predictedOnSurfaceMomentum",
                                &m_predictedOnSurfaceMomentum);
  m_trackTree->SetBranchAddress("filteredOnSurfaceMomentum",
                                &m_filteredOnSurfaceMomentum);
  m_trackTree->SetBranchAddress("smoothedOnSurfaceMomentum",
                                &m_smoothedOnSurfaceMomentum);

  // KF residuals with respect to the true hits
  m_trackTree->SetBranchAddress("truePredictedHitResiduals",
                                &m_truePredictedHitResiduals);
  m_trackTree->SetBranchAddress("trueFilteredHitResiduals",
                                &m_trueFilteredHitResiduals);
  m_trackTree->SetBranchAddress("trueSmoothedHitResiduals",
                                &m_trueSmoothedHitResiduals);

  m_trackTree->SetBranchAddress("truePredictedAngleResiduals",
                                &m_truePredictedAngleResiduals);
  m_trackTree->SetBranchAddress("trueFilteredAngleResiduals",
                                &m_trueFilteredAngleResiduals);
  m_trackTree->SetBranchAddress("trueSmoothedAngleResiduals",
                                &m_trueSmoothedAngleResiduals);

  m_trackTree->SetBranchAddress("trueLeaveOneOutHitResiduals",
                                &m_trueLeaveOneOutHitResiduals);

  // KF residuals with respect to the measurements
  m_trackTree->SetBranchAddress("predictedHitResiduals",
                                &m_predictedHitResiduals);
  m_trackTree->SetBranchAddress("filteredHitResiduals",
                                &m_filteredHitResiduals);
  m_trackTree->SetBranchAddress("smoothedHitResiduals",
                                &m_smoothedHitResiduals);

  m_trackTree->SetBranchAddress("predictedAngleResiduals",
                                &m_predictedAngleResiduals);
  m_trackTree->SetBranchAddress("filteredAngleResiduals",
                                &m_filteredAngleResiduals);
  m_trackTree->SetBranchAddress("smoothedAngleResiduals",
                                &m_smoothedAngleResiduals);

  m_trackTree->SetBranchAddress("leaveOneOutHitResiduals",
                                &m_leaveOneOutHitResiduals);

  // KF pulls with respect to the true hits
  m_trackTree->SetBranchAddress("truePredictedHitPulls",
                                &m_truePredictedHitPulls);
  m_trackTree->SetBranchAddress("trueFilteredHitPulls",
                                &m_trueFilteredHitPulls);
  m_trackTree->SetBranchAddress("trueSmoothedHitPulls",
                                &m_trueSmoothedHitPulls);

  m_trackTree->SetBranchAddress("truePredictedAnglePulls",
                                &m_truePredictedAnglePulls);
  m_trackTree->SetBranchAddress("trueFilteredAnglePulls",
                                &m_trueFilteredAnglePulls);
  m_trackTree->SetBranchAddress("trueSmoothedAnglePulls",
                                &m_trueSmoothedAnglePulls);

  m_trackTree->SetBranchAddress("trueLeaveOneOutHitPulls",
                                &m_trueLeaveOneOutHitPulls);

  // KF pulls with respect to the measurements
  m_trackTree->SetBranchAddress("predictedHitPulls", &m_predictedHitPulls);
  m_trackTree->SetBranchAddress("filteredHitPulls", &m_filteredHitPulls);
  m_trackTree->SetBranchAddress("smoothedHitPulls", &m_smoothedHitPulls);

  m_trackTree->SetBranchAddress("predictedAnglePulls", &m_predictedAnglePulls);
  m_trackTree->SetBranchAddress("filteredAnglePulls", &m_filteredAnglePulls);
  m_trackTree->SetBranchAddress("smoothedAnglePulls", &m_smoothedAnglePulls);

  m_trackTree->SetBranchAddress("leaveOneOutHitPulls", &m_leaveOneOutHitPulls);

  // KF states covariances
  m_trackTree->SetBranchAddress("predictedTrackStateCovs",
                                &m_predictedTrackStateCovs);
  m_trackTree->SetBranchAddress("filteredTrackStateCovs",
                                &m_filteredTrackStateCovs);
  m_trackTree->SetBranchAddress("smoothedTrackStateCovs",
                                &m_smoothedTrackStateCovs);
  m_trackTree->SetBranchAddress("leaveOneOutTrackStateCovs",
                                &m_leaveOneOutTrackStateCovs);

  // Guessed bound track parameters
  m_trackTree->SetBranchAddress("boundTrackParametersGuess",
                                &m_boundTrackParametersGuess);
  m_trackTree->SetBranchAddress("boundTrackCovGuess", &m_boundTrackCovGuess);

  // KF predicted bound track parameters
  m_trackTree->SetBranchAddress("boundTrackParametersEst",
                                &m_boundTrackParametersEst);
  m_trackTree->SetBranchAddress("boundTrackCovEst", &m_boundTrackCovEst);

  // True bound track parameters
  m_trackTree->SetBranchAddress("boundTrackParametersTruth",
                                &m_boundTrackParametersTruth);
  m_trackTree->SetBranchAddress("boundTrackCovTruth", &m_boundTrackCovTruth);

  // Initial guess of the momentum at the origin
  m_trackTree->SetBranchAddress("originMomentumGuess", &m_originMomentumGuess);

  // Initial guess of the vertex at the origin
  m_trackTree->SetBranchAddress("vertexGuess", &m_vertexGuess);

  // KF predicted momentum at the origin
  m_trackTree->SetBranchAddress("originMomentumEst", &m_originMomentumEst);

  // KF predicted vertex at the origin
  m_trackTree->SetBranchAddress("vertexEst", &m_vertexEst);

  // True momentum at the origin
  m_trackTree->SetBranchAddress("originMomentumTruth", &m_originMomentumTruth);

  // True vertex at the origin
  m_trackTree->SetBranchAddress("vertexTruth", &m_vertexTruth);

  // Chi2 of the track with respect ot the measurement
  m_trackTree->SetBranchAddress("chi2Predicted", &m_chi2Predicted);
  m_trackTree->SetBranchAddress("chi2Filtered", &m_chi2Filtered);
  m_trackTree->SetBranchAddress("chi2Smoothed", &m_chi2Smoothed);

  // Number of degrees of freedom of the track
  m_trackTree->SetBranchAddress("ndf", &m_ndf);

  // Track IDs of track states' sim clusters
  m_trackTree->SetBranchAddress("stateTrackId", &m_stateTrackId);
  m_trackTree->SetBranchAddress("stateParentTrackId", &m_stateParentTrackId);
  m_trackTree->SetBranchAddress("stateRunId", &m_stateRunId);

  // Track IDs of the track selected as the majority among the clusters' track
  // IDs
  m_trackTree->SetBranchAddress("trackId", &m_trackId);
  m_trackTree->SetBranchAddress("parentTrackId", &m_parentTrackId);
  m_trackTree->SetBranchAddress("runId", &m_runId);

  // True number of clusters in the track
  m_trackTree->SetBranchAddress("trueTrackSize", &m_trueTrackSize);

  // Number of captured clusters in the track
  m_trackTree->SetBranchAddress("capturedTrackSize", &m_capturedTrackSize);

  // PDG ID
  m_trackTree->SetBranchAddress("pdgId", &m_pdgId);

  // Charge
  m_trackTree->SetBranchAddress("charge", &m_charge);

  // Event ID
  m_trackTree->SetBranchAddress("eventId", &m_eventId);

  // HT cell intersections count
  m_trackTree->SetBranchAddress("xCount", &m_xCount);
}

void SimRootTrackAnalyzer::process(const Options &opt) {
  Stats stats;
  stats.init(opt.prefix, opt.statsOpt);

  std::unordered_map<std::size_t, std::size_t> clustersPerEvent;
  std::unordered_map<std::size_t, std::unordered_map<std::size_t, std::size_t>>
      clustersPerEventPerChip;
  for (std::size_t i = 0; i < m_clusterTree->GetEntries(); i++) {
    m_clusterTree->GetEntry(i);
    clustersPerEvent[m_clusterEventId] += 1 / opt.statsOpt.chipIds.size();
    clustersPerEventPerChip[m_clusterEventId][m_geoId]++;
  }
  std::cout << "FILLED CLUSTERS DATA\n";

  std::size_t nEntries = m_trackTree->GetEntries();
  std::unordered_map<std::size_t, std::size_t> tracksPerEvent;
  for (std::size_t i = 0; i < nEntries; i++) {
    m_trackTree->GetEntry(i);

    // Counters and cut flow
    stats.nTotal++;
    std::cout << "GETTING CLUSTERS FOR " << m_eventId << " "
              << clustersPerEvent.contains(m_eventId) << "\n";
    if (clustersPerEvent.at(m_eventId) < opt.minOcc ||
        clustersPerEvent.at(m_eventId) > opt.maxOcc) {
      continue;
    }
    stats.nPassedOccupancy++;
    if (m_capturedTrackSize < opt.minCapturedTrackSize ||
        m_capturedTrackSize > opt.maxCapturedTrackSize) {
      continue;
    }
    stats.nPassedCapturedTracksSize++;
    if (m_xCount < opt.minXCount || m_xCount > opt.maxXCount) {
      continue;
    }
    stats.nPassedXCount++;
    if (m_chi2Predicted < opt.minChi2Predicted ||
        m_chi2Predicted > opt.maxChi2Predicted) {
      continue;
    }
    stats.nPassedChi2Predicted++;
    if (m_chi2Filtered < opt.minChi2Filtered ||
        m_chi2Filtered > opt.maxChi2Filtered) {
      continue;
    }
    stats.nPassedChi2Filtered++;
    if (m_chi2Smoothed < opt.minChi2Smoothed ||
        m_chi2Smoothed > opt.maxChi2Smoothed) {
      continue;
    }
    stats.nPassedChi2Smoothed++;
    if (m_vertexEst->Z() < opt.minVertexZ ||
        m_vertexEst->Z() > opt.maxVertexZ) {
      continue;
    }
    stats.nPassedVertexZ++;
    if (m_vertexEst->Y() < opt.minVertexY ||
        m_vertexEst->Y() > opt.maxVertexY) {
      continue;
    }
    stats.nPassedVertexY++;
    if (m_originMomentumEst->Px() < opt.minMomentumX ||
        m_originMomentumEst->Px() > opt.maxMomentumX) {
      continue;
    }
    stats.nPassedMomentumZ++;

    // bool passed = true;
    // for (std::size_t j = 0; j < m_smoothedHitResiduals->size(); j++) {
    //   std::size_t geoId = m_geometryIds->at(j);
    //   const auto &smoothedRes = m_smoothedHitResiduals->at(j);
    //   if (geoId == 16) {
    //     if (smoothedRes.X() < 0.02 || smoothedRes.X() > 0.06 ||
    //         smoothedRes.Y() < -0.08 || smoothedRes.Y() > -0.02) {
    //       passed = false;
    //       break;
    //     }
    //   }
    // }
    // if (!passed) {
    //   continue;
    // }
    stats.nPassedResiduals++;

    tracksPerEvent[m_eventId]++;

    stats.capturedTrackSizeHist.Fill(m_capturedTrackSize);
    stats.xCountHist.Fill(m_xCount);

    stats.magnetConfHist.Fill(m_dipoleStrength, m_xCorrectorStrength);

    // Chi2
    stats.chi2PredictedHist.Fill(m_chi2Predicted);
    stats.chi2FilteredHist.Fill(m_chi2Filtered);
    stats.chi2SmoothedHist.Fill(m_chi2Smoothed);

    /// Chi2 functions
    stats.chi2SmoothedCapturedTrackSizeHist.Fill(m_chi2Smoothed,
                                                 m_capturedTrackSize);
    stats.chi2SmoothedXCountHist.Fill(m_chi2Smoothed, m_xCount);

    std::unordered_map<std::size_t, double> hitXX;
    std::unordered_map<std::size_t, double> hitYY;

    std::unordered_map<std::size_t, double> resXX;
    std::unordered_map<std::size_t, double> resYY;
    std::cout << "ITERATIONG\n";
    for (std::size_t j = 0; j < m_smoothedHitPulls->size(); j++) {
      std::size_t geoId = m_geometryIds->at(j);
      std::cout << "GEO ID " << geoId << "\n";

      std::cout << "GETTING CLUSTERS\n";
      stats.nClustersHist.at(geoId).Fill(
          clustersPerEventPerChip.at(m_eventId).at(geoId));
      std::cout << "GOT CLUSTERS\n";

      // Residuals
      std::cout << "GETTING RESIDUALS\n";
      const auto &predictedRes = m_predictedHitResiduals->at(j);
      const auto &filteredRes = m_filteredHitResiduals->at(j);
      const auto &smoothedRes = m_smoothedHitResiduals->at(j);
      const auto &leaveOneOutRes = m_leaveOneOutHitResiduals->at(j);
      std::cout << "GOT RESIDUALS\n";

      stats.predictedResidualsXCumulativeHist.Fill(predictedRes.X());
      stats.filteredResidualsXCumulativeHist.Fill(filteredRes.X());
      stats.smoothedResidualsXCumulativeHist.Fill(smoothedRes.X());

      stats.predictedResidualsYCumulativeHist.Fill(predictedRes.Y());
      stats.filteredResidualsYCumulativeHist.Fill(filteredRes.Y());
      stats.smoothedResidualsYCumulativeHist.Fill(smoothedRes.Y());

      stats.filteredResidualsXYHists.at(geoId).Fill(filteredRes.X(),
                                                    filteredRes.Y());
      stats.smoothedResidualsXYHists.at(geoId).Fill(smoothedRes.X(),
                                                    smoothedRes.Y());
      stats.leaveOneOutResidualsXYHists.at(geoId).Fill(leaveOneOutRes.X(),
                                                       leaveOneOutRes.Y());

      resXX[geoId] = smoothedRes.X();
      resYY[geoId] = smoothedRes.Y();

      stats.predictedResidualsXHists.at(geoId).Fill(predictedRes.X());
      stats.filteredResidualsXHists.at(geoId).Fill(filteredRes.X());
      stats.smoothedResidualsXHists.at(geoId).Fill(smoothedRes.X());
      stats.leaveOneOutResidualsXHists.at(geoId).Fill(leaveOneOutRes.X());

      stats.predictedResidualsYHists.at(geoId).Fill(predictedRes.Y());
      stats.filteredResidualsYHists.at(geoId).Fill(filteredRes.Y());
      stats.smoothedResidualsYHists.at(geoId).Fill(smoothedRes.Y());
      stats.leaveOneOutResidualsYHists.at(geoId).Fill(leaveOneOutRes.Y());

      // Pulls
      const auto &predictedPull = m_predictedHitPulls->at(j);
      const auto &filteredPull = m_filteredHitPulls->at(j);
      const auto &smoothedPull = m_smoothedHitPulls->at(j);
      const auto &leaveOneOutPull = m_leaveOneOutHitPulls->at(j);

      stats.predictedPullsXCumulativeHist.Fill(predictedPull.X());
      stats.filteredPullsXCumulativeHist.Fill(filteredPull.X());
      stats.smoothedPullsXCumulativeHist.Fill(smoothedPull.X());

      stats.predictedPullsYCumulativeHist.Fill(predictedPull.Y());
      stats.filteredPullsYCumulativeHist.Fill(filteredPull.Y());
      stats.smoothedPullsYCumulativeHist.Fill(smoothedPull.Y());

      stats.predictedPullsXHists.at(geoId).Fill(predictedPull.X());
      stats.filteredPullsXHists.at(geoId).Fill(filteredPull.X());
      stats.smoothedPullsXHists.at(geoId).Fill(smoothedPull.X());
      stats.leaveOneOutPullsXHists.at(geoId).Fill(leaveOneOutPull.X());

      stats.predictedPullsYHists.at(geoId).Fill(predictedPull.Y());
      stats.filteredPullsYHists.at(geoId).Fill(filteredPull.Y());
      stats.smoothedPullsYHists.at(geoId).Fill(smoothedPull.Y());
      stats.leaveOneOutPullsYHists.at(geoId).Fill(leaveOneOutPull.Y());

      // Local track hits
      const auto &trackHitLocal = m_trackHitsLocal->at(j);
      stats.trackHitsLocalHists.at(geoId).Fill(trackHitLocal.X(),
                                               trackHitLocal.Y());

      hitXX[geoId] = trackHitLocal.X();
      hitYY[geoId] = trackHitLocal.Y();

      // Track hits errors
      const auto &trackHitCov = m_trackHitCovs->at(j);
      double hitErrorX = std::sqrt(trackHitCov(0, 0));
      double hitErrorY = std::sqrt(trackHitCov(1, 1));
      stats.trackHitsErrorXHists.at(geoId).Fill(hitErrorX);
      stats.trackHitsErrorYHists.at(geoId).Fill(hitErrorY);
      stats.trackHitsErrorXYHists.at(geoId).Fill(hitErrorX, hitErrorY);

      // Track residuals vs track hit parameters
      auto &rxhxHist = stats.smoothedResidualsXTrackHitsLocalXHists.at(geoId);
      auto &rxhyHist = stats.smoothedResidualsXTrackHitsLocalYHists.at(geoId);
      auto &rxhphiHist =
          stats.smoothedResidualsXTrackHitsLocalPhiHists.at(geoId);
      auto &ryhxHist = stats.smoothedResidualsYTrackHitsLocalXHists.at(geoId);
      auto &ryhyHist = stats.smoothedResidualsYTrackHitsLocalYHists.at(geoId);
      auto &ryhphiHist =
          stats.smoothedResidualsYTrackHitsLocalPhiHists.at(geoId);
      auto &rxhexHist = stats.smoothedResidualsXTrackHitsErrorXHists.at(geoId);
      auto &ryheyHist = stats.smoothedResidualsYTrackHitsErrorYHists.at(geoId);

      rxhxHist.Fill(trackHitLocal.X(), smoothedRes.X());
      rxhyHist.Fill(trackHitLocal.Y(), smoothedRes.X());
      rxhphiHist.Fill(std::atan(trackHitLocal.Y() / trackHitLocal.X()),
                      smoothedRes.X());
      ryhxHist.Fill(trackHitLocal.X(), smoothedRes.Y());
      ryhyHist.Fill(trackHitLocal.Y(), smoothedRes.Y());
      ryhphiHist.Fill(std::atan(trackHitLocal.Y() / trackHitLocal.X()),
                      smoothedRes.Y());
      rxhexHist.Fill(hitErrorX, smoothedRes.X());
      ryheyHist.Fill(hitErrorY, smoothedRes.Y());

      // Track residuals vs track parameters
      auto &rxvxHist = stats.smoothedResidualsXVertexEstXHists.at(geoId);
      auto &rxvyHist = stats.smoothedResidualsXVertexEstYHists.at(geoId);
      auto &ryvxHist = stats.smoothedResidualsYVertexEstXHists.at(geoId);
      auto &ryvyHist = stats.smoothedResidualsYVertexEstYHists.at(geoId);

      auto &rxpyHist =
          stats.smoothedResidualsXOriginMomentumEstPyHists.at(geoId);
      auto &rypyHist =
          stats.smoothedResidualsYOriginMomentumEstPyHists.at(geoId);

      rxvxHist.Fill(-m_vertexEst->Z(), smoothedRes.X());
      rxvyHist.Fill(m_vertexEst->Y(), smoothedRes.X());
      ryvxHist.Fill(-m_vertexEst->Z(), smoothedRes.Y());
      ryvyHist.Fill(m_vertexEst->Y(), smoothedRes.Y());

      rxpyHist.Fill(m_originMomentumEst->Py(), smoothedRes.X());
      rypyHist.Fill(m_originMomentumEst->Py(), smoothedRes.Y());

      // Track pulls vs track hit parameters
      auto &pxhxHist = stats.smoothedPullsXTrackHitsLocalXHists.at(geoId);
      auto &pxhyHist = stats.smoothedPullsXTrackHitsLocalYHists.at(geoId);
      auto &pyhxHist = stats.smoothedPullsYTrackHitsLocalXHists.at(geoId);
      auto &pyhyHist = stats.smoothedPullsYTrackHitsLocalYHists.at(geoId);

      pxhxHist.Fill(trackHitLocal.X(), smoothedPull.X());
      pxhyHist.Fill(trackHitLocal.Y(), smoothedPull.X());
      pyhxHist.Fill(trackHitLocal.X(), smoothedPull.Y());
      pyhyHist.Fill(trackHitLocal.Y(), smoothedPull.Y());
    }
    std::cout << "FILLED TRACK DATA\n";

    std::vector<std::size_t> chipIds_(opt.statsOpt.chipIds.begin(),
                                      opt.statsOpt.chipIds.end());
    std::sort(chipIds_.begin(), chipIds_.end());
    for (std::size_t n = 0; n < chipIds_.size() - 1; n++) {
      std::size_t idN = chipIds_.at(n);
      for (std::size_t m = n + 1; m < chipIds_.size(); m++) {
        std::size_t idM = chipIds_.at(m);

        stats.trackHitsLocalXXHists.at(idN * 100 + idM)
            .Fill(hitXX.at(idN), hitXX.at(idM));
        stats.trackHitsLocalYYHists.at(idN * 100 + idM)
            .Fill(hitYY.at(idN), hitYY.at(idM));

        stats.smoothedResidualsXXHists.at(idN * 100 + idM)
            .Fill(resXX.at(idN), resXX.at(idM));
        stats.smoothedResidualsYYHists.at(idN * 100 + idM)
            .Fill(resYY.at(idN), resYY.at(idM));
      }
    }

    // Momentum
    stats.originMomentumEstPxHist.Fill(-m_originMomentumEst->Pz());
    stats.originMomentumEstPyHist.Fill(m_originMomentumEst->Py());
    stats.originMomentumEstPzHist.Fill(m_originMomentumEst->Px());

    // Momentum direction
    stats.originMomentumEstDxHist.Fill(-m_originMomentumEst->Pz() /
                                       m_originMomentumEst->P());
    stats.originMomentumEstDyHist.Fill(m_originMomentumEst->Py() /
                                       m_originMomentumEst->P());
    stats.originMomentumEstDzHist.Fill(m_originMomentumEst->Px() /
                                       m_originMomentumEst->P());

    stats.originMomentumEstPhiHist.Fill(m_originMomentumEst->Phi());
    stats.originMomentumEstThetaHist.Fill(m_originMomentumEst->Theta());

    // Vertex
    stats.vertexEstXYHist.Fill(-m_vertexEst->Z(), m_vertexEst->Y());
    stats.vertexEstXHist.Fill(-m_vertexEst->Z());
    stats.vertexEstYHist.Fill(m_vertexEst->Y());
    if (i % 10000 == 0) {
      std::cout << i << "/" << nEntries << "\n";
    }
  }

  for (const auto &[id, nTracks] : tracksPerEvent) {
    stats.nTracksHist.Fill(nTracks);
    stats.nTracksNClustersHist.Fill(clustersPerEvent.at(id), nTracks);
  }

  stats.cutFlowHist.Fill("nTotal",
                         static_cast<double>(stats.nTotal) / opt.nBXs);
  stats.cutFlowHist.Fill(
      "nPassedOccupancy",
      static_cast<double>(stats.nPassedOccupancy) / opt.nBXs);
  stats.cutFlowHist.Fill(
      "nPassedCapturedTracksSize",
      static_cast<double>(stats.nPassedCapturedTracksSize) / opt.nBXs);
  stats.cutFlowHist.Fill("nPassedXCount",
                         static_cast<double>(stats.nPassedXCount) / opt.nBXs);
  stats.cutFlowHist.Fill(
      "nPassedChi2Predicted",
      static_cast<double>(stats.nPassedChi2Predicted) / opt.nBXs);
  stats.cutFlowHist.Fill(
      "nPassedChi2Filtered",
      static_cast<double>(stats.nPassedChi2Filtered) / opt.nBXs);
  stats.cutFlowHist.Fill(
      "nPassedChi2Smoothed",
      static_cast<double>(stats.nPassedChi2Smoothed) / opt.nBXs);
  stats.cutFlowHist.Fill("nPassedVertexZ",
                         static_cast<double>(stats.nPassedVertexZ) / opt.nBXs);
  stats.cutFlowHist.Fill("nPassedVertexY",
                         static_cast<double>(stats.nPassedVertexY) / opt.nBXs);
  stats.cutFlowHist.Fill(
      "nPassedMomentumZ",
      static_cast<double>(stats.nPassedMomentumZ) / opt.nBXs);
  stats.cutFlowHist.Fill(
      "nPassedResiduals",
      static_cast<double>(stats.nPassedResiduals) / opt.nBXs);

  // Normalize
  stats.capturedTrackSizeHist.Scale(1.0 / opt.nBXs);

  stats.magnetConfHist.Scale(1.0 / opt.nBXs);

  stats.nTracksHist.Scale(1.0 / opt.nBXs);
  stats.nTracksNClustersHist.Scale(1.0 / opt.nBXs);

  stats.chi2PredictedHist.Scale(1.0 / opt.nBXs);
  stats.chi2FilteredHist.Scale(1.0 / opt.nBXs);
  stats.chi2SmoothedHist.Scale(1.0 / opt.nBXs);

  stats.predictedResidualsXCumulativeHist.Scale(
      1.0 / stats.predictedResidualsXCumulativeHist.Integral(), "width");
  stats.filteredResidualsXCumulativeHist.Scale(
      1.0 / stats.filteredResidualsXCumulativeHist.Integral(), "width");
  stats.smoothedResidualsXCumulativeHist.Scale(
      1.0 / stats.smoothedResidualsXCumulativeHist.Integral(), "width");

  stats.predictedResidualsYCumulativeHist.Scale(
      1.0 / stats.predictedResidualsYCumulativeHist.Integral(), "width");
  stats.filteredResidualsYCumulativeHist.Scale(
      1.0 / stats.filteredResidualsYCumulativeHist.Integral(), "width");
  stats.smoothedResidualsYCumulativeHist.Scale(
      1.0 / stats.smoothedResidualsYCumulativeHist.Integral(), "width");

  stats.predictedPullsXCumulativeHist.Scale(
      1.0 / stats.predictedPullsXCumulativeHist.Integral(), "width");
  stats.filteredPullsXCumulativeHist.Scale(
      1.0 / stats.filteredPullsXCumulativeHist.Integral(), "width");
  stats.smoothedPullsXCumulativeHist.Scale(
      1.0 / stats.smoothedPullsXCumulativeHist.Integral(), "width");

  stats.predictedPullsYCumulativeHist.Scale(
      1.0 / stats.predictedPullsYCumulativeHist.Integral(), "width");
  stats.filteredPullsYCumulativeHist.Scale(
      1.0 / stats.filteredPullsYCumulativeHist.Integral(), "width");
  stats.smoothedPullsYCumulativeHist.Scale(
      1.0 / stats.smoothedPullsYCumulativeHist.Integral(), "width");

  for (std::size_t j = 0; j < m_smoothedHitPulls->size(); j++) {
    std::size_t geoId = m_geometryIds->at(j);

    stats.nClustersHist.at(geoId).Scale(1.0 / opt.nBXs);

    stats.trackHitsLocalHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.trackHitsErrorXHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.trackHitsErrorYHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.trackHitsErrorXYHists.at(geoId).Scale(1.0 / opt.nBXs);

    stats.leaveOneOutTrackHitsErrorXHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.leaveOneOutTrackHitsErrorYHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.leaveOneOutTrackHitsErrorXYHists.at(geoId).Scale(1.0 / opt.nBXs);

    stats.looTrackHitsErrorModelTrackHitsErrorXHists.at(geoId).Scale(1.0 /
                                                                     opt.nBXs);
    stats.looTrackHitsErrorModelTrackHitsErrorYHists.at(geoId).Scale(1.0 /
                                                                     opt.nBXs);

    stats.looTrackHitsErrorXTrackHitsLocalXHists.at(geoId).Scale(1.0 /
                                                                 opt.nBXs);

    stats.filteredResidualsXYHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.smoothedResidualsXYHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.leaveOneOutResidualsXYHists.at(geoId).Scale(1.0 / opt.nBXs);

    stats.predictedResidualsXHists.at(geoId).Scale(
        1.0 / stats.predictedResidualsXHists.at(geoId).Integral(), "width");
    stats.filteredResidualsXHists.at(geoId).Scale(
        1.0 / stats.filteredResidualsXHists.at(geoId).Integral(), "width");
    stats.smoothedResidualsXHists.at(geoId).Scale(
        1.0 / stats.smoothedResidualsXHists.at(geoId).Integral(), "width");
    stats.leaveOneOutResidualsXHists.at(geoId).Scale(
        1.0 / stats.leaveOneOutResidualsXHists.at(geoId).Integral(), "width");

    stats.predictedResidualsYHists.at(geoId).Scale(
        1.0 / stats.predictedResidualsYHists.at(geoId).Integral(), "width");
    stats.filteredResidualsYHists.at(geoId).Scale(
        1.0 / stats.filteredResidualsYHists.at(geoId).Integral(), "width");
    stats.smoothedResidualsYHists.at(geoId).Scale(
        1.0 / stats.smoothedResidualsYHists.at(geoId).Integral(), "width");
    stats.leaveOneOutResidualsYHists.at(geoId).Scale(
        1.0 / stats.leaveOneOutResidualsYHists.at(geoId).Integral(), "width");

    stats.predictedPullsXHists.at(geoId).Scale(
        1.0 / stats.predictedPullsXHists.at(geoId).Integral(), "width");
    stats.filteredPullsXHists.at(geoId).Scale(
        1.0 / stats.filteredPullsXHists.at(geoId).Integral(), "width");
    stats.smoothedPullsXHists.at(geoId).Scale(
        1.0 / stats.smoothedPullsXHists.at(geoId).Integral(), "width");
    stats.leaveOneOutPullsXHists.at(geoId).Scale(
        1.0 / stats.leaveOneOutPullsXHists.at(geoId).Integral(), "width");

    stats.predictedPullsYHists.at(geoId).Scale(
        1.0 / stats.predictedPullsYHists.at(geoId).Integral(), "width");
    stats.filteredPullsYHists.at(geoId).Scale(
        1.0 / stats.filteredPullsYHists.at(geoId).Integral(), "width");
    stats.smoothedPullsYHists.at(geoId).Scale(
        1.0 / stats.smoothedPullsYHists.at(geoId).Integral(), "width");
    stats.leaveOneOutPullsYHists.at(geoId).Scale(
        1.0 / stats.leaveOneOutPullsYHists.at(geoId).Integral(), "width");

    stats.smoothedResidualsXTrackHitsLocalXHists.at(geoId).Scale(1.0 /
                                                                 opt.nBXs);
    stats.smoothedResidualsXTrackHitsLocalYHists.at(geoId).Scale(1.0 /
                                                                 opt.nBXs);
    stats.smoothedResidualsYTrackHitsLocalXHists.at(geoId).Scale(1.0 /
                                                                 opt.nBXs);
    stats.smoothedResidualsYTrackHitsLocalYHists.at(geoId).Scale(1.0 /
                                                                 opt.nBXs);
    stats.smoothedResidualsXTrackHitsLocalPhiHists.at(geoId).Scale(1.0 /
                                                                   opt.nBXs);
    stats.smoothedResidualsYTrackHitsLocalPhiHists.at(geoId).Scale(1.0 /
                                                                   opt.nBXs);
    stats.smoothedResidualsXTrackHitsErrorXHists.at(geoId).Scale(1.0 /
                                                                 opt.nBXs);
    stats.smoothedResidualsYTrackHitsErrorYHists.at(geoId).Scale(1.0 /
                                                                 opt.nBXs);

    stats.smoothedResidualsXVertexEstXHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.smoothedResidualsXVertexEstYHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.smoothedResidualsYVertexEstXHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.smoothedResidualsYVertexEstYHists.at(geoId).Scale(1.0 / opt.nBXs);

    stats.smoothedResidualsXOriginMomentumEstPyHists.at(geoId).Scale(1.0 /
                                                                     opt.nBXs);
    stats.smoothedResidualsYOriginMomentumEstPyHists.at(geoId).Scale(1.0 /
                                                                     opt.nBXs);

    stats.smoothedPullsXTrackHitsLocalXHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.smoothedPullsXTrackHitsLocalYHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.smoothedPullsYTrackHitsLocalXHists.at(geoId).Scale(1.0 / opt.nBXs);
    stats.smoothedPullsYTrackHitsLocalYHists.at(geoId).Scale(1.0 / opt.nBXs);
  }
  for (auto &[id, hist] : stats.trackHitsLocalXXHists) {
    hist.Scale(1.0 / opt.nBXs);
  }
  for (auto &[id, hist] : stats.trackHitsLocalYYHists) {
    hist.Scale(1.0 / opt.nBXs);
  }
  for (auto &[id, hist] : stats.smoothedResidualsXXHists) {
    hist.Scale(1.0 / opt.nBXs);
  }
  for (auto &[id, hist] : stats.smoothedResidualsYYHists) {
    hist.Scale(1.0 / opt.nBXs);
  }

  stats.originMomentumEstPxHist.Scale(1.0 / opt.nBXs);
  stats.originMomentumEstPyHist.Scale(1.0 / opt.nBXs);
  stats.originMomentumEstPzHist.Scale(1.0 / opt.nBXs);

  stats.originMomentumEstDxHist.Scale(1.0 / opt.nBXs);
  stats.originMomentumEstDyHist.Scale(1.0 / opt.nBXs);
  stats.originMomentumEstDzHist.Scale(1.0 / opt.nBXs);

  stats.originMomentumEstPhiHist.Scale(1.0 / opt.nBXs);
  stats.originMomentumEstThetaHist.Scale(1.0 / opt.nBXs);

  stats.vertexEstXYHist.Scale(1.0 / opt.nBXs);
  stats.vertexEstXHist.Scale(1.0 / opt.nBXs);
  stats.vertexEstYHist.Scale(1.0 / opt.nBXs);

  // Print stats
  auto [probErrLeft, probErrRight] = getBinomialErrors(
      stats.nPassedVertexY, stats.nTotal - stats.nPassedVertexY, 1e-5);
  std::cout << "Prob: "
            << static_cast<double>(stats.nPassedVertexY) / stats.nTotal << " + "
            << probErrRight << " - " << probErrLeft << "\n";
  std::cout << "Tracks: " << stats.nPassedVertexY << " + "
            << stats.nTotal * probErrRight << " - "
            << stats.nTotal * probErrLeft << "\n";
  std::cout << "Rate: " << static_cast<double>(stats.nPassedVertexY) / opt.nBXs
            << " + " << stats.nTotal * probErrRight / opt.nBXs << " - "
            << stats.nTotal * probErrLeft / opt.nBXs << "\n";

  stats.print();
}

void SimRootTrackAnalyzer::Stats::init(const std::string &pfix,
                                       const Options &opt) {
  prefix = pfix;
  chipIds = opt.chipIds;

  // Cut flow
  std::string nameCutFlow = prefix + "CutFlow";
  cutFlowHist = TH1D(nameCutFlow.c_str(), nameCutFlow.c_str(), 6, 0, 6);
  cutFlowHist.Fill("nTotal", 0);
  cutFlowHist.Fill("nPassedCapturedTracksSize", 0);
  cutFlowHist.Fill("nPassedXCount", 0);
  cutFlowHist.Fill("nPassedChi2Predicted", 0);
  cutFlowHist.Fill("nPassedChi2Filtered", 0);
  cutFlowHist.Fill("nPassedChi2Smoothed", 0);
  cutFlowHist.Fill("nPassedVertexZ", 0);
  cutFlowHist.Fill("nPassedVertexY", 0);
  cutFlowHist.Fill("nPassedMomentumZ", 0);
  cutFlowHist.Fill("nPassedResiduals", 0);

  // Captured track size
  std::string nameCapturedTrackSize = prefix + "CapturedTrackSize";
  capturedTrackSizeHist =
      TH1D(nameCapturedTrackSize.c_str(), nameCapturedTrackSize.c_str(),
           opt.capturedTrackSizeBins, opt.capturedTrackSizeMin,
           opt.capturedTrackSizeMax);

  // X count
  std::string nameXCount = prefix + "XCount";
  xCountHist = TH1D(nameXCount.c_str(), nameXCount.c_str(), opt.xCountBins,
                    opt.xCountMin, opt.xCountMax);

  // Mag conf hist
  std::string nameMagConf = prefix + "MagConf";
  magnetConfHist = TH2D(nameMagConf.c_str(), nameMagConf.c_str(),
                        opt.dipoleStrengthBins, opt.dipoleStrengthMin,
                        opt.dipoleStrengthMax, opt.xCorrectorStrengthBins,
                        opt.xCorrectorStrengthMin, opt.xCorrectorStrengthMax);

  // Number of tracks/BX vs number of clusters/BX
  for (int id : chipIds) {
    std::string nameNClusters = prefix + "NClusters" + std::to_string(id);
    nClustersHist.insert(
        {id, TH1D(nameNClusters.c_str(), nameNClusters.c_str(),
                  opt.nClustersBins, opt.nClustersMin, opt.nClustersMax)});
  }
  std::string nameNTracks = prefix + "NTracks";
  nTracksHist = TH1D(nameNTracks.c_str(), nameNTracks.c_str(), opt.nTracksBins,
                     opt.nTracksMin, opt.nTracksMax);
  std::string nameNTracksNClusters = prefix + "NTracksNClusters";
  nTracksNClustersHist =
      TH2D(nameNTracksNClusters.c_str(), nameNTracksNClusters.c_str(),
           opt.nClustersBins, opt.nClustersMin, opt.nClustersMax,
           opt.nTracksBins, opt.nTracksMin, opt.nTracksMax);

  // Chi2
  std::string nameChi2PredictedHist = prefix + "Chi2Predicted";
  chi2PredictedHist =
      TH1D(nameChi2PredictedHist.c_str(), nameChi2PredictedHist.c_str(),
           opt.chi2Bins, opt.chi2Min, opt.chi2Max);
  std::string nameChi2FilteredHist = prefix + "Chi2Filtered";
  chi2FilteredHist =
      TH1D(nameChi2FilteredHist.c_str(), nameChi2FilteredHist.c_str(),
           opt.chi2Bins, opt.chi2Min, opt.chi2Max);
  std::string nameChi2SmoothedHist = prefix + "Chi2Smoothed";
  chi2SmoothedHist =
      TH1D(nameChi2SmoothedHist.c_str(), nameChi2SmoothedHist.c_str(),
           opt.chi2Bins, opt.chi2Min, opt.chi2Max);

  /// Chi2 functions
  std::string nameChi2SmoothedXCountHist = prefix + "Chi2SmoothedXCount";
  chi2SmoothedXCountHist =
      TH2D(nameChi2SmoothedXCountHist.c_str(),
           nameChi2SmoothedXCountHist.c_str(), opt.chi2Bins, opt.chi2Min,
           opt.chi2Max, opt.xCountBins, opt.xCountMin, opt.xCountMax);

  // Local track hits
  for (int id : chipIds) {
    std::string nameTrackHitsLocal =
        prefix + "TrackHitsLocal" + std::to_string(id);
    trackHitsLocalHists.insert(
        {id, TH2D(nameTrackHitsLocal.c_str(), nameTrackHitsLocal.c_str(),
                  opt.trackHitsLocalBinsX, opt.trackHitsLocalMinX,
                  opt.trackHitsLocalMaxX, opt.trackHitsLocalBinsY,
                  opt.trackHitsLocalMinY, opt.trackHitsLocalMaxY)});
  }
  std::vector<std::size_t> chipIds_(chipIds.begin(), chipIds.end());
  std::sort(chipIds_.begin(), chipIds_.end());
  for (std::size_t n = 0; n < chipIds_.size() - 1; n++) {
    std::size_t idN = chipIds_.at(n);
    for (std::size_t m = n + 1; m < chipIds_.size(); m++) {
      auto mit = chipIds_.begin();
      std::advance(mit, m);
      std::size_t idM = chipIds_.at(m);

      std::string nameTrackHitsLocalXXHists = prefix + "TrackHitsLocalXX" +
                                              std::to_string(idN) +
                                              std::to_string(idM);
      trackHitsLocalXXHists.insert(
          {idN * 100 + idM,
           TH2D(nameTrackHitsLocalXXHists.c_str(),
                nameTrackHitsLocalXXHists.c_str(), opt.trackHitsLocalBinsX,
                opt.trackHitsLocalMinX, opt.trackHitsLocalMaxX,
                opt.trackHitsLocalBinsX, opt.trackHitsLocalMinX,
                opt.trackHitsLocalMaxX)});
      std::string nameTrackHitsLocalYYHists = prefix + "TrackHitsLocalYY" +
                                              std::to_string(idN) +
                                              std::to_string(idM);
      trackHitsLocalYYHists.insert(
          {idN * 100 + idM,
           TH2D(nameTrackHitsLocalYYHists.c_str(),
                nameTrackHitsLocalYYHists.c_str(), opt.trackHitsLocalBinsY,
                opt.trackHitsLocalMinY, opt.trackHitsLocalMaxY,
                opt.trackHitsLocalBinsY, opt.trackHitsLocalMinY,
                opt.trackHitsLocalMaxY)});
    }
  }

  // Track hits errors
  for (int id : chipIds) {
    std::string nameTrackHitsErrorX =
        prefix + "TrackHitsErrorX" + std::to_string(id);
    trackHitsErrorXHists.insert(
        {id, TH1D(nameTrackHitsErrorX.c_str(), nameTrackHitsErrorX.c_str(),
                  opt.trackHitsErrorBinsX, opt.trackHitsErrorMinX,
                  opt.trackHitsErrorMaxX)});
    std::string nameTrackHitsErrorY =
        prefix + "TrackHitsErrorY" + std::to_string(id);
    trackHitsErrorYHists.insert(
        {id, TH1D(nameTrackHitsErrorY.c_str(), nameTrackHitsErrorY.c_str(),
                  opt.trackHitsErrorBinsX, opt.trackHitsErrorMinX,
                  opt.trackHitsErrorMaxX)});
    std::string nameTrackHitsErrorXY =
        prefix + "TrackHitsErrorXY" + std::to_string(id);
    trackHitsErrorXYHists.insert(
        {id, TH2D(nameTrackHitsErrorXY.c_str(), nameTrackHitsErrorXY.c_str(),
                  opt.trackHitsErrorBinsX, opt.trackHitsErrorMinX,
                  opt.trackHitsErrorMaxX, opt.trackHitsErrorBinsY,
                  opt.trackHitsErrorMinY, opt.trackHitsErrorMaxY)});

    std::string nameLeaveOneOutTrackHitsErrorX =
        prefix + "LeaveOneOutTrackHitsErrorX" + std::to_string(id);
    leaveOneOutTrackHitsErrorXHists.insert(
        {id, TH1D(nameLeaveOneOutTrackHitsErrorX.c_str(),
                  nameLeaveOneOutTrackHitsErrorX.c_str(),
                  opt.trackHitsLooErrorBinsX, opt.trackHitsLooErrorMinX,
                  opt.trackHitsLooErrorMaxX)});
    std::string nameLeaveOneOutTrackHitsErrorY =
        prefix + "LeaveOneOutTrackHitsErrorY" + std::to_string(id);
    leaveOneOutTrackHitsErrorYHists.insert(
        {id, TH1D(nameLeaveOneOutTrackHitsErrorY.c_str(),
                  nameLeaveOneOutTrackHitsErrorY.c_str(),
                  opt.trackHitsLooErrorBinsY, opt.trackHitsLooErrorMinY,
                  opt.trackHitsLooErrorMaxY)});
    std::string nameLeaveOneOutTrackHitsErrorXY =
        prefix + "LeaveOneOutTrackHitsErrorXY" + std::to_string(id);
    leaveOneOutTrackHitsErrorXYHists.insert(
        {id, TH2D(nameLeaveOneOutTrackHitsErrorXY.c_str(),
                  nameLeaveOneOutTrackHitsErrorXY.c_str(),
                  opt.trackHitsLooErrorBinsX, opt.trackHitsLooErrorMinX,
                  opt.trackHitsLooErrorMaxX, opt.trackHitsLooErrorBinsY,
                  opt.trackHitsLooErrorMinY, opt.trackHitsLooErrorMaxY)});

    std::string nameLooTrackHitsErrorModelTrackHitsErrorX =
        prefix + "LooTrackHitsErrorModelTrackHitsErrorX" + std::to_string(id);
    looTrackHitsErrorModelTrackHitsErrorXHists.insert(
        {id, TH2D(nameLooTrackHitsErrorModelTrackHitsErrorX.c_str(),
                  nameLooTrackHitsErrorModelTrackHitsErrorX.c_str(),
                  opt.trackHitsLooErrorBinsX, opt.trackHitsLooErrorMinX,
                  opt.trackHitsLooErrorMaxX, opt.trackHitsErrorBinsX,
                  opt.trackHitsErrorMinX, opt.trackHitsErrorMaxX)});
    std::string nameLooTrackHitsErrorModelTrackHitsErrorY =
        prefix + "LooTrackHitsErrorModelTrackHitsErrorY" + std::to_string(id);
    looTrackHitsErrorModelTrackHitsErrorYHists.insert(
        {id, TH2D(nameLooTrackHitsErrorModelTrackHitsErrorY.c_str(),
                  nameLooTrackHitsErrorModelTrackHitsErrorY.c_str(),
                  opt.trackHitsLooErrorBinsY, opt.trackHitsLooErrorMinY,
                  opt.trackHitsLooErrorMaxY, opt.trackHitsErrorBinsY,
                  opt.trackHitsErrorMinY, opt.trackHitsErrorMaxY)});

    std::string nameLooTrackHitsErrorXTrackHitsLocalX =
        prefix + "LooTrackHitsErrorXTrackHitsLocalX" + std::to_string(id);
    looTrackHitsErrorXTrackHitsLocalXHists.insert(
        {id, TH2D(nameLooTrackHitsErrorXTrackHitsLocalX.c_str(),
                  nameLooTrackHitsErrorXTrackHitsLocalX.c_str(),
                  opt.trackHitsLooErrorBinsX, opt.trackHitsLooErrorMinX,
                  opt.trackHitsLooErrorMaxX, opt.trackHitsLocalBinsX,
                  opt.trackHitsLocalMinX, opt.trackHitsLocalMaxX)});
  }

  // Residuals
  std::string namePredictedResidualsXCumulativeHist =
      prefix + "PredictedResidualsXCumulative";
  predictedResidualsXCumulativeHist =
      TH1D(namePredictedResidualsXCumulativeHist.c_str(),
           namePredictedResidualsXCumulativeHist.c_str(), opt.residualsXBins,
           opt.residualsXMin, opt.residualsXMax);
  std::string nameFilteredResidualsXCumulativeHist =
      prefix + "FilteredResidualsXCumulative";
  filteredResidualsXCumulativeHist =
      TH1D(nameFilteredResidualsXCumulativeHist.c_str(),
           nameFilteredResidualsXCumulativeHist.c_str(), opt.residualsXBins,
           opt.residualsXMin, opt.residualsXMax);
  std::string nameSmoothedResidualsXCumulativeHist =
      prefix + "SmoothedResidualsXCumulative";
  smoothedResidualsXCumulativeHist =
      TH1D(nameSmoothedResidualsXCumulativeHist.c_str(),
           nameSmoothedResidualsXCumulativeHist.c_str(), opt.residualsXBins,
           opt.residualsXMin, opt.residualsXMax);

  std::string namePredictedResidualsYCumulativeHist =
      prefix + "PredictedResidualsYCumulative";
  predictedResidualsYCumulativeHist =
      TH1D(namePredictedResidualsYCumulativeHist.c_str(),
           namePredictedResidualsYCumulativeHist.c_str(), opt.residualsYBins,
           opt.residualsYMin, opt.residualsYMax);
  std::string nameFilteredResidualsYCumulativeHist =
      prefix + "FilteredResidualsYCumulative";
  filteredResidualsYCumulativeHist =
      TH1D(nameFilteredResidualsYCumulativeHist.c_str(),
           nameFilteredResidualsYCumulativeHist.c_str(), opt.residualsYBins,
           opt.residualsYMin, opt.residualsYMax);
  std::string nameSmoothedResidualsYCumulativeHist =
      prefix + "SmoothedResidualsYCumulative";
  smoothedResidualsYCumulativeHist =
      TH1D(nameSmoothedResidualsYCumulativeHist.c_str(),
           nameSmoothedResidualsYCumulativeHist.c_str(), opt.residualsYBins,
           opt.residualsYMin, opt.residualsYMax);

  for (int id : chipIds) {
    std::string nameFilteredResidualsXYHists =
        prefix + "FilteredResidualsXY" + std::to_string(id);
    filteredResidualsXYHists.insert(
        {id, TH2D(nameFilteredResidualsXYHists.c_str(),
                  nameFilteredResidualsXYHists.c_str(), opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax, opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});
    std::string nameSmoothedResidualsXYHists =
        prefix + "SmoothedResidualsXY" + std::to_string(id);
    smoothedResidualsXYHists.insert(
        {id, TH2D(nameSmoothedResidualsXYHists.c_str(),
                  nameSmoothedResidualsXYHists.c_str(), opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax, opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});
    std::string nameLeaveOneOutResidualsXYHists =
        prefix + "LeaveOneOutResidualsXY" + std::to_string(id);
    leaveOneOutResidualsXYHists.insert(
        {id,
         TH2D(nameLeaveOneOutResidualsXYHists.c_str(),
              nameLeaveOneOutResidualsXYHists.c_str(), opt.looResidualsXBins,
              opt.looResidualsXMin, opt.looResidualsXMax, opt.looResidualsYBins,
              opt.looResidualsYMin, opt.looResidualsYMax)});

    std::string namePredictedResidualsXHists =
        prefix + "PredictedResidualsX" + std::to_string(id);
    predictedResidualsXHists.insert(
        {id, TH1D(namePredictedResidualsXHists.c_str(),
                  namePredictedResidualsXHists.c_str(), opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax)});
    std::string nameFilteredResidualsXHists =
        prefix + "FilteredResidualsX" + std::to_string(id);
    filteredResidualsXHists.insert(
        {id, TH1D(nameFilteredResidualsXHists.c_str(),
                  nameFilteredResidualsXHists.c_str(), opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax)});
    std::string nameSmoothedResidualsXHists =
        prefix + "SmoothedResidualsX" + std::to_string(id);
    smoothedResidualsXHists.insert(
        {id, TH1D(nameSmoothedResidualsXHists.c_str(),
                  nameSmoothedResidualsXHists.c_str(), opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax)});
    std::string nameLeaveOneOutResidualsXHists =
        prefix + "LeaveOneOutResidualsX" + std::to_string(id);
    leaveOneOutResidualsXHists.insert(
        {id, TH1D(nameLeaveOneOutResidualsXHists.c_str(),
                  nameLeaveOneOutResidualsXHists.c_str(), opt.looResidualsXBins,
                  opt.looResidualsXMin, opt.looResidualsXMax)});

    std::string namePredictedResidualsYHists =
        prefix + "PredictedResidualsY" + std::to_string(id);
    predictedResidualsYHists.insert(
        {id, TH1D(namePredictedResidualsYHists.c_str(),
                  namePredictedResidualsYHists.c_str(), opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});
    std::string nameFilteredResidualsYHists =
        prefix + "FilteredResidualsY" + std::to_string(id);
    filteredResidualsYHists.insert(
        {id, TH1D(nameFilteredResidualsYHists.c_str(),
                  nameFilteredResidualsYHists.c_str(), opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});
    std::string nameSmoothedResidualsYHists =
        prefix + "SmoothedResidualsY" + std::to_string(id);
    smoothedResidualsYHists.insert(
        {id, TH1D(nameSmoothedResidualsYHists.c_str(),
                  nameSmoothedResidualsYHists.c_str(), opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});
    std::string nameLeaveOneOutResidualsYHists =
        prefix + "LeaveOneOutResidualsY" + std::to_string(id);
    leaveOneOutResidualsYHists.insert(
        {id, TH1D(nameLeaveOneOutResidualsYHists.c_str(),
                  nameLeaveOneOutResidualsYHists.c_str(), opt.looResidualsYBins,
                  opt.looResidualsYMin, opt.looResidualsYMax)});
  }
  for (std::size_t n = 0; n < chipIds_.size() - 1; n++) {
    std::size_t idN = chipIds_.at(n);
    for (std::size_t m = n + 1; m < chipIds_.size(); m++) {
      auto mit = chipIds_.begin();
      std::advance(mit, m);
      std::size_t idM = chipIds_.at(m);

      std::string nameSmoothedResidualsXXHists =
          prefix + "SmoothedResidualsXX" + std::to_string(idN) +
          std::to_string(idM);
      smoothedResidualsXXHists.insert(
          {idN * 100 + idM,
           TH2D(nameSmoothedResidualsXXHists.c_str(),
                nameSmoothedResidualsXXHists.c_str(), opt.residualsXBins,
                opt.residualsXMin, opt.residualsXMax, opt.residualsXBins,
                opt.residualsXMin, opt.residualsXMax)});
      std::string nameSmoothedResidualsYYHists =
          prefix + "SmoothedResidualsYY" + std::to_string(idN) +
          std::to_string(idM);
      smoothedResidualsYYHists.insert(
          {idN * 100 + idM,
           TH2D(nameSmoothedResidualsYYHists.c_str(),
                nameSmoothedResidualsYYHists.c_str(), opt.residualsYBins,
                opt.residualsYMin, opt.residualsYMax, opt.residualsYBins,
                opt.residualsYMin, opt.residualsYMax)});
    }
  }

  // Pulls
  std::string namePredictedPullsXCumulativeHist =
      prefix + "PredictedPullsXCumulative";
  predictedPullsXCumulativeHist =
      TH1D(namePredictedPullsXCumulativeHist.c_str(),
           namePredictedPullsXCumulativeHist.c_str(), opt.pullsXBins,
           opt.pullsXMin, opt.pullsXMax);
  std::string nameFilteredPullsXCumulativeHist =
      prefix + "FilteredPullsXCumulative";
  filteredPullsXCumulativeHist =
      TH1D(nameFilteredPullsXCumulativeHist.c_str(),
           nameFilteredPullsXCumulativeHist.c_str(), opt.pullsXBins,
           opt.pullsXMin, opt.pullsXMax);
  std::string nameSmoothedPullsXCumulativeHist =
      prefix + "SmoothedPullsXCumulative";
  smoothedPullsXCumulativeHist =
      TH1D(nameSmoothedPullsXCumulativeHist.c_str(),
           nameSmoothedPullsXCumulativeHist.c_str(), opt.pullsXBins,
           opt.pullsXMin, opt.pullsXMax);

  std::string namePredictedPullsYCumulativeHist =
      prefix + "PredictedPullsYCumulative";
  predictedPullsYCumulativeHist =
      TH1D(namePredictedPullsYCumulativeHist.c_str(),
           namePredictedPullsYCumulativeHist.c_str(), opt.pullsYBins,
           opt.pullsYMin, opt.pullsYMax);
  std::string nameFilteredPullsYCumulativeHist =
      prefix + "FilteredPullsYCumulative";
  filteredPullsYCumulativeHist =
      TH1D(nameFilteredPullsYCumulativeHist.c_str(),
           nameFilteredPullsYCumulativeHist.c_str(), opt.pullsYBins,
           opt.pullsYMin, opt.pullsYMax);
  std::string nameSmoothedPullsYCumulativeHist =
      prefix + "SmoothedPullsYCumulative";
  smoothedPullsYCumulativeHist =
      TH1D(nameSmoothedPullsYCumulativeHist.c_str(),
           nameSmoothedPullsYCumulativeHist.c_str(), opt.pullsYBins,
           opt.pullsYMin, opt.pullsYMax);

  for (int id : chipIds) {
    std::string namePredictedPullsXHists =
        prefix + "PredictedPullsX" + std::to_string(id);
    predictedPullsXHists.insert(
        {id, TH1D(namePredictedPullsXHists.c_str(),
                  namePredictedPullsXHists.c_str(), opt.pullsXBins,
                  opt.pullsXMin, opt.pullsXMax)});
    std::string nameFilteredPullsXHists =
        prefix + "FilteredPullsX" + std::to_string(id);
    filteredPullsXHists.insert(
        {id,
         TH1D(nameFilteredPullsXHists.c_str(), nameFilteredPullsXHists.c_str(),
              opt.pullsXBins, opt.pullsXMin, opt.pullsXMax)});
    std::string nameSmoothedPullsXHists =
        prefix + "SmoothedPullsX" + std::to_string(id);
    smoothedPullsXHists.insert(
        {id,
         TH1D(nameSmoothedPullsXHists.c_str(), nameSmoothedPullsXHists.c_str(),
              opt.pullsXBins, opt.pullsXMin, opt.pullsXMax)});
    std::string nameLeaveOneOutPullsXHists =
        prefix + "LeaveOneOutPullsX" + std::to_string(id);
    leaveOneOutPullsXHists.insert(
        {id, TH1D(nameLeaveOneOutPullsXHists.c_str(),
                  nameLeaveOneOutPullsXHists.c_str(), opt.pullsXBins,
                  opt.pullsXMin, opt.pullsXMax)});

    std::string namePredictedPullsYHists =
        prefix + "PredictedPullsY" + std::to_string(id);
    predictedPullsYHists.insert(
        {id, TH1D(namePredictedPullsYHists.c_str(),
                  namePredictedPullsYHists.c_str(), opt.pullsYBins,
                  opt.pullsYMin, opt.pullsYMax)});
    std::string nameFilteredPullsYHists =
        prefix + "FilteredPullsY" + std::to_string(id);
    filteredPullsYHists.insert(
        {id,
         TH1D(nameFilteredPullsYHists.c_str(), nameFilteredPullsYHists.c_str(),
              opt.pullsYBins, opt.pullsYMin, opt.pullsYMax)});
    std::string nameSmoothedPullsYHists =
        prefix + "SmoothedPullsY" + std::to_string(id);
    smoothedPullsYHists.insert(
        {id,
         TH1D(nameSmoothedPullsYHists.c_str(), nameSmoothedPullsYHists.c_str(),
              opt.pullsYBins, opt.pullsYMin, opt.pullsYMax)});
    std::string nameLeaveOneOutPullsYHists =
        prefix + "LeaveOneOutPullsY" + std::to_string(id);
    leaveOneOutPullsYHists.insert(
        {id, TH1D(nameLeaveOneOutPullsYHists.c_str(),
                  nameLeaveOneOutPullsYHists.c_str(), opt.pullsYBins,
                  opt.pullsYMin, opt.pullsYMax)});
  }

  // Momentum
  std::string nameOriginMomentumEstPxHist = prefix + "OriginMomentumEstPx";
  originMomentumEstPxHist = TH1D(
      nameOriginMomentumEstPxHist.c_str(), nameOriginMomentumEstPxHist.c_str(),
      opt.momentumXBins, opt.momentumXMin, opt.momentumXMax);
  std::string nameOriginMomentumEstPyHist = prefix + "OriginMomentumEstPy";
  originMomentumEstPyHist = TH1D(
      nameOriginMomentumEstPyHist.c_str(), nameOriginMomentumEstPyHist.c_str(),
      opt.momentumYBins, opt.momentumYMin, opt.momentumYMax);
  std::string nameOriginMomentumEstPzHist = prefix + "OriginMomentumEstPz";
  originMomentumEstPzHist = TH1D(
      nameOriginMomentumEstPzHist.c_str(), nameOriginMomentumEstPzHist.c_str(),
      opt.momentumZBins, opt.momentumZMin, opt.momentumZMax);

  // Momentum direction
  std::string nameOriginMomentumEstDxHist = prefix + "OriginMomentumEstDx";
  originMomentumEstDxHist = TH1D(
      nameOriginMomentumEstDxHist.c_str(), nameOriginMomentumEstDxHist.c_str(),
      opt.momentumDirXBins, opt.momentumDirXMin, opt.momentumDirXMax);
  std::string nameOriginMomentumEstDyHist = prefix + "OriginMomentumEstDy";
  originMomentumEstDyHist = TH1D(
      nameOriginMomentumEstDyHist.c_str(), nameOriginMomentumEstDyHist.c_str(),
      opt.momentumDirYBins, opt.momentumDirYMin, opt.momentumDirYMax);
  std::string nameOriginMomentumEstDzHist = prefix + "OriginMomentumEstDz";
  originMomentumEstDzHist = TH1D(
      nameOriginMomentumEstDzHist.c_str(), nameOriginMomentumEstDzHist.c_str(),
      opt.momentumDirZBins, opt.momentumDirZMin, opt.momentumDirZMax);

  std::string nameOriginMomentumEstPhiHist = prefix + "OriginMomentumEstPhi";
  originMomentumEstPhiHist =
      TH1D(nameOriginMomentumEstPhiHist.c_str(),
           nameOriginMomentumEstPhiHist.c_str(), opt.momentumPhiBins,
           opt.momentumPhiMin, opt.momentumPhiMax);
  std::string nameOriginMomentumEstThetaHist =
      prefix + "OriginMomentumEstTheta";
  originMomentumEstThetaHist =
      TH1D(nameOriginMomentumEstThetaHist.c_str(),
           nameOriginMomentumEstThetaHist.c_str(), opt.momentumThetaBins,
           opt.momentumThetaMin, opt.momentumThetaMax);

  // Vertex
  std::string nameVertexEstXYHist = prefix + "VertexEstXY";
  vertexEstXYHist =
      TH2D(nameVertexEstXYHist.c_str(), nameVertexEstXYHist.c_str(),
           opt.vertexXBins, opt.vertexXMin, opt.vertexXMax, opt.vertexYBins,
           opt.vertexYMin, opt.vertexYMax);
  std::string nameVertexEstXHist = prefix + "VertexEstX";
  vertexEstXHist = TH1D(nameVertexEstXHist.c_str(), nameVertexEstXHist.c_str(),
                        opt.vertexXBins, opt.vertexXMin, opt.vertexXMax);
  std::string nameVertexEstYHist = prefix + "VertexEstY";
  vertexEstYHist = TH1D(nameVertexEstYHist.c_str(), nameVertexEstYHist.c_str(),
                        opt.vertexYBins, opt.vertexYMin, opt.vertexYMax);

  // Track residuals vs track hit parameters
  for (int id : chipIds) {
    std::string rxtxName =
        prefix + "SmoothedResidualsXTrackHitsLocalX" + std::to_string(id);
    smoothedResidualsXTrackHitsLocalXHists.insert(
        {id, TH2D(rxtxName.c_str(), rxtxName.c_str(), opt.trackHitsLocalBinsX,
                  opt.trackHitsLocalMinX, opt.trackHitsLocalMaxX,
                  opt.residualsXBins, opt.residualsXMin, opt.residualsXMax)});
    std::string rxtyName =
        prefix + "SmoothedResidualsXTrackHitsLocalY" + std::to_string(id);
    smoothedResidualsXTrackHitsLocalYHists.insert(
        {id, TH2D(rxtyName.c_str(), rxtyName.c_str(), opt.trackHitsLocalBinsY,
                  opt.trackHitsLocalMinY, opt.trackHitsLocalMaxY,
                  opt.residualsXBins, opt.residualsXMin, opt.residualsXMax)});
    std::string rxtphiName =
        prefix + "SmoothedResidualsXTrackHitsLocalPhi" + std::to_string(id);
    smoothedResidualsXTrackHitsLocalPhiHists.insert(
        {id, TH2D(rxtphiName.c_str(), rxtphiName.c_str(), 100, -M_PI, M_PI,
                  opt.residualsXBins, opt.residualsXMin, opt.residualsXMax)});
    std::string rytxName =
        prefix + "SmoothedResidualsYTrackHitsLocalX" + std::to_string(id);
    smoothedResidualsYTrackHitsLocalXHists.insert(
        {id, TH2D(rytxName.c_str(), rytxName.c_str(), opt.trackHitsLocalBinsX,
                  opt.trackHitsLocalMinX, opt.trackHitsLocalMaxX,
                  opt.residualsYBins, opt.residualsYMin, opt.residualsYMax)});
    std::string rytyName =
        prefix + "SmoothedResidualsYTrackHitsLocalY" + std::to_string(id);
    smoothedResidualsYTrackHitsLocalYHists.insert(
        {id, TH2D(rytyName.c_str(), rytyName.c_str(), opt.trackHitsLocalBinsY,
                  opt.trackHitsLocalMinY, opt.trackHitsLocalMaxY,
                  opt.residualsYBins, opt.residualsYMin, opt.residualsYMax)});
    std::string rytphiName =
        prefix + "SmoothedResidualsYTrackHitsLocalPhi" + std::to_string(id);
    smoothedResidualsYTrackHitsLocalPhiHists.insert(
        {id, TH2D(rytphiName.c_str(), rytphiName.c_str(), 100, -M_PI, M_PI,
                  opt.residualsYBins, opt.residualsYMin, opt.residualsYMax)});
    std::string rxhexName =
        prefix + "SmoothedResidualsXTrackHitsErrorX" + std::to_string(id);
    smoothedResidualsXTrackHitsErrorXHists.insert(
        {id, TH2D(rxhexName.c_str(), rxhexName.c_str(), opt.trackHitsErrorBinsX,
                  opt.trackHitsErrorMinX, opt.trackHitsErrorMaxX,
                  opt.residualsXBins, opt.residualsXMin, opt.residualsXMax)});
    std::string ryheyName =
        prefix + "SmoothedResidualsYTrackHitsErrorY" + std::to_string(id);
    smoothedResidualsYTrackHitsErrorYHists.insert(
        {id, TH2D(ryheyName.c_str(), ryheyName.c_str(), opt.trackHitsErrorBinsY,
                  opt.trackHitsErrorMinY, opt.trackHitsErrorMaxY,
                  opt.residualsYBins, opt.residualsYMin, opt.residualsYMax)});

    std::string rxvxName =
        prefix + "SmoothedResidualsXVertexEstX" + std::to_string(id);
    smoothedResidualsXVertexEstXHists.insert(
        {id, TH2D(rxvxName.c_str(), rxvxName.c_str(), opt.vertexXBins,
                  opt.vertexXMin, opt.vertexXMax, opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax)});
    std::string rxvyName =
        prefix + "SmoothedResidualsXVertexEstY" + std::to_string(id);
    smoothedResidualsXVertexEstYHists.insert(
        {id, TH2D(rxvyName.c_str(), rxvyName.c_str(), opt.vertexYBins,
                  opt.vertexYMin, opt.vertexYMax, opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax)});
    std::string ryvxName =
        prefix + "SmoothedResidualsYVertexEstX" + std::to_string(id);
    smoothedResidualsYVertexEstXHists.insert(
        {id, TH2D(ryvxName.c_str(), ryvxName.c_str(), opt.vertexXBins,
                  opt.vertexXMin, opt.vertexXMax, opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});
    std::string ryvyName =
        prefix + "SmoothedResidualsYVertexEstY" + std::to_string(id);
    smoothedResidualsYVertexEstYHists.insert(
        {id, TH2D(ryvyName.c_str(), ryvyName.c_str(), opt.vertexYBins,
                  opt.vertexYMin, opt.vertexYMax, opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});

    std::string rxpyName =
        prefix + "SmoothedResidualsXOriginMomentumEstPy" + std::to_string(id);
    smoothedResidualsXOriginMomentumEstPyHists.insert(
        {id, TH2D(rxpyName.c_str(), rxpyName.c_str(), opt.momentumYBins,
                  opt.momentumYMin, opt.momentumYMax, opt.residualsXBins,
                  opt.residualsXMin, opt.residualsXMax)});
    std::string rypyName =
        prefix + "SmoothedResidualsYOriginMomentumEstPy" + std::to_string(id);
    smoothedResidualsYOriginMomentumEstPyHists.insert(
        {id, TH2D(rypyName.c_str(), rypyName.c_str(), opt.momentumYBins,
                  opt.momentumYMin, opt.momentumYMax, opt.residualsYBins,
                  opt.residualsYMin, opt.residualsYMax)});
  }

  // Track pulls vs track hit parameters
  for (int id : chipIds) {
    std::string pxtxName =
        prefix + "SmoothedPullsXTrackHitsLocalX" + std::to_string(id);
    smoothedPullsXTrackHitsLocalXHists.insert(
        {id, TH2D(pxtxName.c_str(), pxtxName.c_str(), opt.trackHitsLocalBinsX,
                  opt.trackHitsLocalMinX, opt.trackHitsLocalMaxX,
                  opt.pullsXBins, opt.pullsXMin, opt.pullsXMax)});
    std::string pxtyName =
        prefix + "SmoothedPullsXTrackHitsLocalY" + std::to_string(id);
    smoothedPullsXTrackHitsLocalYHists.insert(
        {id, TH2D(pxtyName.c_str(), pxtyName.c_str(), opt.trackHitsLocalBinsY,
                  opt.trackHitsLocalMinY, opt.trackHitsLocalMaxY,
                  opt.pullsXBins, opt.pullsXMin, opt.pullsXMax)});
    std::string pytxName =
        prefix + "SmoothedPullsYTrackHitsLocalX" + std::to_string(id);
    smoothedPullsYTrackHitsLocalXHists.insert(
        {id, TH2D(pytxName.c_str(), pytxName.c_str(), opt.trackHitsLocalBinsX,
                  opt.trackHitsLocalMinX, opt.trackHitsLocalMaxX,
                  opt.pullsYBins, opt.pullsYMin, opt.pullsYMax)});
    std::string pytyName =
        prefix + "SmoothedPullsYTrackHitsLocalY" + std::to_string(id);
    smoothedPullsYTrackHitsLocalYHists.insert(
        {id, TH2D(pytyName.c_str(), pytyName.c_str(), opt.trackHitsLocalBinsY,
                  opt.trackHitsLocalMinY, opt.trackHitsLocalMaxY,
                  opt.pullsYBins, opt.pullsYMin, opt.pullsYMax)});
  }
}

void SimRootTrackAnalyzer::Stats::reset() {}

void SimRootTrackAnalyzer::Stats::print() {
  using enum HistPainterConfig::StatBoxConfig::EntryOpt;

  TCanvas c = TCanvas(prefix.c_str(), "", 800, 800);
  TH1D dummy("dummy", "dummy", 100, 0, 100);

  LegendPainterConfig legCfg;
  legCfg.draw = true;
  legCfg.option = "p";

  std::string cOptionFirst = prefix + ".pdf(";
  std::string cOptionLast = prefix + ".pdf)";
  std::string cOption = prefix + ".pdf";

  // FuncPainterConfig fitFuncCfg;
  // fitFuncCfg.lineWidth = 2;
  // fitFuncCfg.drawPars = true;
  // fitFuncCfg.parsTextSize = 0.036;
  // fitFuncCfg.parsStartX = 0.0035;
  // fitFuncCfg.parsStartY = 95;
  // fitFuncCfg.parsSpacingY = 10;
  // fitFuncCfg.parsScale = 1e3;
  // fitFuncCfg.parsUnits = {"um", "um"};

  // Counters
  HistPainterConfig cfgCutFlow;
  cfgCutFlow.yTitle = "Tracks/BX";
  cfgCutFlow.option = "hist text";

  saveHist(c, cOptionFirst, {cutFlowHist, cfgCutFlow});

  HistPainterConfig cfgCapturedTrackSize;
  cfgCapturedTrackSize.canvasXSize = 10000;
  cfgCapturedTrackSize.canvasYSize = 10000;
  cfgCapturedTrackSize.stats = true;
  cfgCapturedTrackSize.xTitle = "Captured track size, [N clusters]";
  cfgCapturedTrackSize.yTitle = "Tracks/BX";
  cfgCapturedTrackSize.padLeftMargin = 0.14;
  cfgCapturedTrackSize.padRightMargin = 0.06;
  cfgCapturedTrackSize.yTitleOffset = 1.5;

  saveHist(c, cOption, {capturedTrackSizeHist, cfgCapturedTrackSize});

  HistPainterConfig cfgXCount;
  cfgXCount.canvasXSize = 10000;
  cfgXCount.canvasYSize = 10000;
  cfgXCount.stats = true;
  cfgXCount.xTitle = "X count, [NX]";
  cfgXCount.yTitle = "Tracks/BX";
  cfgXCount.padLeftMargin = 0.14;
  cfgXCount.padRightMargin = 0.06;
  cfgXCount.yTitleOffset = 1.5;

  saveHist(c, cOption, {xCountHist, cfgXCount});

  HistPainterConfig cfgMagConf;
  cfgMagConf.canvasXSize = 10000;
  cfgMagConf.canvasYSize = 10000.;
  cfgMagConf.xTitle = "Dipole strength, [T]";
  cfgMagConf.yTitle = "XCOR strength, [T]";
  cfgMagConf.padLeftMargin = 0.14;
  cfgMagConf.padRightMargin = 0.14;
  cfgMagConf.yTitleOffset = 1.5;
  cfgMagConf.option = "colz";

  saveHist(c, cOptionFirst, {magnetConfHist, cfgMagConf});

  HistPainterConfig cfgNTracks;
  cfgNTracks.canvasXSize = 10000;
  cfgNTracks.canvasYSize = 10000.;
  cfgNTracks.stats = true;
  cfgNTracks.xTitle = "N tracks/BX";
  cfgNTracks.yTitle = "Entries";
  cfgNTracks.padLeftMargin = 0.14;
  cfgNTracks.padRightMargin = 0.14;
  cfgNTracks.yTitleOffset = 1.5;

  HistPainterConfig cfgNClusters;
  cfgNClusters.canvasXSize = 10000;
  cfgNClusters.canvasYSize = 10000. / chipIds.size();
  cfgNClusters.stats = true;
  cfgNClusters.xTitle = "N clusters/BX";
  cfgNClusters.yTitle = "Entries";
  cfgNClusters.padLeftMargin = 0.14;
  cfgNClusters.padRightMargin = 0.14;
  cfgNClusters.yTitleOffset = 1.5;

  HistPainterConfig cfgNTracksNClusters;
  cfgNTracksNClusters.canvasXSize = 10000;
  cfgNTracksNClusters.canvasYSize = 10000.;
  cfgNTracksNClusters.xTitle = "N clusters/(chip * BX)";
  cfgNTracksNClusters.yTitle = "N tracks/BX";
  cfgNTracksNClusters.padLeftMargin = 0.14;
  cfgNTracksNClusters.padRightMargin = 0.14;
  cfgNTracksNClusters.yTitleOffset = 1.5;
  cfgNTracksNClusters.option = "colz";

  std::vector<std::pair<TH1 &, HistPainterConfig>> ncHists;
  for (std::size_t id : chipIds) {
    ncHists.push_back({nClustersHist.at(id), cfgNClusters});
  }

  saveHist(c, cOption, ncHists);
  saveHist(c, cOption, {nTracksHist, cfgNTracks});
  saveHist(c, cOption, {nTracksNClustersHist, cfgNTracksNClusters});

  // Chi2
  HistPainterConfig cfgChi2;
  cfgChi2.canvasXSize = 10000;
  cfgChi2.canvasYSize = 10000. / 3;
  cfgChi2.stats = true;
  cfgChi2.xTitle = "#chi^{2}";
  cfgChi2.yTitle = "Tracks/BX";
  cfgChi2.padLeftMargin = 0.14;
  cfgChi2.padRightMargin = 0.06;
  cfgChi2.yTitleOffset = 1.5;

  saveHist(c, cOption,
           {{chi2PredictedHist, cfgChi2},
            {chi2FilteredHist, cfgChi2},
            {chi2SmoothedHist, cfgChi2}});

  /// Chi2 functions
  HistPainterConfig cfgCapturedTrackSizeChi2;
  cfgCapturedTrackSizeChi2.canvasXSize = 10000;
  cfgCapturedTrackSizeChi2.canvasYSize = 10000;
  cfgCapturedTrackSizeChi2.stats = true;
  cfgCapturedTrackSizeChi2.xTitle = "#chi^{2}";
  cfgCapturedTrackSizeChi2.yTitle = "Captured track size, [N clusters]";
  cfgCapturedTrackSizeChi2.padLeftMargin = 0.14;
  cfgCapturedTrackSizeChi2.padRightMargin = 0.06;
  cfgCapturedTrackSizeChi2.yTitleOffset = 1.5;

  saveHist(c, cOptionFirst,
           {chi2SmoothedCapturedTrackSizeHist, cfgCapturedTrackSizeChi2});

  HistPainterConfig cfgXCountChi2;
  cfgXCountChi2.canvasXSize = 10000;
  cfgXCountChi2.canvasYSize = 10000;
  cfgXCountChi2.stats = true;
  cfgXCountChi2.xTitle = "#chi^{2}";
  cfgXCountChi2.yTitle = "X count, [NX]";
  cfgXCountChi2.padLeftMargin = 0.14;
  cfgXCountChi2.padRightMargin = 0.06;
  cfgXCountChi2.yTitleOffset = 1.5;

  saveHist(c, cOptionFirst, {chi2SmoothedXCountHist, cfgXCountChi2});

  // Local track hits
  HistPainterConfig cfgLocalTrackHits;
  cfgLocalTrackHits.canvasXSize = 10000;
  cfgLocalTrackHits.canvasYSize = 10000. / chipIds.size();
  cfgLocalTrackHits.stats = false;
  cfgLocalTrackHits.xTitle = "x_{hit}, [mm]";
  cfgLocalTrackHits.yTitle = "y_{hit}, [mm]";
  cfgLocalTrackHits.option = "col";
  std::vector<std::pair<TH1 &, HistPainterConfig>> hxhyHists;
  for (std::size_t id : chipIds) {
    hxhyHists.push_back({trackHitsLocalHists.at(id), cfgLocalTrackHits});
  }
  saveHist(c, cOption, hxhyHists);

  HistPainterConfig cfgHitXHitX;
  cfgHitXHitX.canvasXSize = 10000;
  cfgHitXHitX.canvasYSize = 10000. / (chipIds.size() - 1);
  cfgHitXHitX.stats = true;
  cfgHitXHitX.xTitle = "x^{hit}, [mm]";
  cfgHitXHitX.yTitle = "x^{hit}, [mm]";
  cfgHitXHitX.xAdjustDivisions = true;
  cfgHitXHitX.xNDivisions = 5;

  HistPainterConfig cfgHitYHitY;
  cfgHitYHitY.canvasXSize = 10000;
  cfgHitYHitY.canvasYSize = 10000. / (chipIds.size() - 1);
  cfgHitYHitY.stats = true;
  cfgHitYHitY.xTitle = "y^{hit}, [mm]";
  cfgHitYHitY.yTitle = "y^{hit}, [mm]";
  cfgHitYHitY.xAdjustDivisions = true;
  cfgHitYHitY.xNDivisions = 5;

  std::vector<std::size_t> chipIds_(chipIds.begin(), chipIds.end());
  std::sort(chipIds_.begin(), chipIds_.end());
  for (std::size_t n = 0; n < chipIds_.size() - 1; n++) {
    std::size_t idN = chipIds_.at(n);
    std::vector<std::pair<TH1 &, HistPainterConfig>> hitxhitxHists;
    for (std::size_t m = n + 1; m < chipIds_.size(); m++) {
      std::size_t idM = chipIds_.at(m);
      hitxhitxHists.push_back(
          {trackHitsLocalXXHists.at(idN * 100 + idM), cfgHitXHitX});
    }
    for (std::size_t m = hitxhitxHists.size(); m < chipIds_.size() - 1; m++) {
      hitxhitxHists.push_back({dummy, cfgHitXHitX});
    }
    saveHist(c, cOption, hitxhitxHists);
  }
  for (std::size_t n = 0; n < chipIds_.size() - 1; n++) {
    std::size_t idN = chipIds_.at(n);
    std::vector<std::pair<TH1 &, HistPainterConfig>> hityhityHists;
    for (std::size_t m = n + 1; m < chipIds_.size(); m++) {
      std::size_t idM = chipIds_.at(m);
      hityhityHists.push_back(
          {trackHitsLocalYYHists.at(idN * 100 + idM), cfgHitYHitY});
    }
    for (std::size_t m = hityhityHists.size(); m < chipIds_.size() - 1; m++) {
      hityhityHists.push_back({dummy, cfgHitYHitY});
    }
    saveHist(c, cOption, hityhityHists);
  }

  // Track hits error
  HistPainterConfig cfgTrackHitsErrorX;
  cfgTrackHitsErrorX.canvasXSize = 10000;
  cfgTrackHitsErrorX.canvasYSize = 10000. / chipIds.size();
  cfgTrackHitsErrorX.stats = true;
  cfgTrackHitsErrorX.xTitle = "#sigma_{x_{hit}}, [mm]";
  cfgTrackHitsErrorX.yTitle = "Hits/BX";

  HistPainterConfig cfgTrackHitsErrorY;
  cfgTrackHitsErrorY.canvasXSize = 10000;
  cfgTrackHitsErrorY.canvasYSize = 10000. / chipIds.size();
  cfgTrackHitsErrorY.stats = true;
  cfgTrackHitsErrorY.xTitle = "#sigma_{x_{hit}}, [mm]";
  cfgTrackHitsErrorY.yTitle = "Hits/BX";

  HistPainterConfig cfgTrackHitsErrorXY;
  cfgTrackHitsErrorXY.canvasXSize = 10000;
  cfgTrackHitsErrorXY.canvasYSize = 10000. / chipIds.size();
  cfgTrackHitsErrorXY.stats = false;
  cfgTrackHitsErrorXY.xTitle = "#sigma_{x_{hit}}, [mm]";
  cfgTrackHitsErrorXY.yTitle = "#sigma_{y_{hit}}, [mm]";
  cfgTrackHitsErrorXY.option = "col";

  HistPainterConfig cfgLooTrackHitsErrorModelTrackHitsErrorX;
  cfgLooTrackHitsErrorModelTrackHitsErrorX.canvasXSize = 10000;
  cfgLooTrackHitsErrorModelTrackHitsErrorX.canvasYSize =
      10000. / chipIds.size();
  cfgLooTrackHitsErrorModelTrackHitsErrorX.stats = false;
  cfgLooTrackHitsErrorModelTrackHitsErrorX.xTitle =
      "#sigma_{x_{hit, loo}}, [mm]";
  cfgLooTrackHitsErrorModelTrackHitsErrorX.yTitle =
      "#sigma_{x_{hit, model}}, [mm]";
  cfgLooTrackHitsErrorModelTrackHitsErrorX.option = "col";

  HistPainterConfig cfgLooTrackHitsErrorModelTrackHitsErrorY;
  cfgLooTrackHitsErrorModelTrackHitsErrorY.canvasXSize = 10000;
  cfgLooTrackHitsErrorModelTrackHitsErrorY.canvasYSize =
      10000. / chipIds.size();
  cfgLooTrackHitsErrorModelTrackHitsErrorY.stats = false;
  cfgLooTrackHitsErrorModelTrackHitsErrorY.xTitle =
      "#sigma_{x_{hit, loo}}, [mm]";
  cfgLooTrackHitsErrorModelTrackHitsErrorY.yTitle =
      "#sigma_{x_{hit, model}}, [mm]";
  cfgLooTrackHitsErrorModelTrackHitsErrorY.option = "col";

  HistPainterConfig cfgLooTrackHitsErrorXTrackHitsLocalX;
  cfgLooTrackHitsErrorXTrackHitsLocalX.canvasXSize = 10000;
  cfgLooTrackHitsErrorXTrackHitsLocalX.canvasYSize = 10000. / chipIds.size();
  cfgLooTrackHitsErrorXTrackHitsLocalX.stats = false;
  cfgLooTrackHitsErrorXTrackHitsLocalX.xTitle = "#sigma_{x_{hit, loo}}, [mm]";
  cfgLooTrackHitsErrorXTrackHitsLocalX.yTitle = "x_{hit}, [mm]";
  cfgLooTrackHitsErrorXTrackHitsLocalX.option = "col";

  std::vector<std::pair<TH1 &, HistPainterConfig>> hexHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> heyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> hexheyHists;

  std::vector<std::pair<TH1 &, HistPainterConfig>> loohexHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> looheyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> loohexheyHists;

  std::vector<std::pair<TH1 &, HistPainterConfig>> loohexmhexHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> looheymheyHists;

  std::vector<std::pair<TH1 &, HistPainterConfig>> loohexthlxHists;
  for (std::size_t id : chipIds) {
    hexHists.push_back({trackHitsErrorXHists.at(id), cfgTrackHitsErrorX});
    heyHists.push_back({trackHitsErrorYHists.at(id), cfgTrackHitsErrorY});
    hexheyHists.push_back({trackHitsErrorXYHists.at(id), cfgTrackHitsErrorXY});

    loohexHists.push_back(
        {leaveOneOutTrackHitsErrorXHists.at(id), cfgTrackHitsErrorX});
    looheyHists.push_back(
        {leaveOneOutTrackHitsErrorYHists.at(id), cfgTrackHitsErrorY});
    loohexheyHists.push_back(
        {leaveOneOutTrackHitsErrorXYHists.at(id), cfgTrackHitsErrorXY});

    loohexmhexHists.push_back(
        {looTrackHitsErrorModelTrackHitsErrorXHists.at(id),
         cfgLooTrackHitsErrorModelTrackHitsErrorX});
    looheymheyHists.push_back(
        {looTrackHitsErrorModelTrackHitsErrorYHists.at(id),
         cfgLooTrackHitsErrorModelTrackHitsErrorY});
    loohexthlxHists.push_back({looTrackHitsErrorXTrackHitsLocalXHists.at(id),
                               cfgLooTrackHitsErrorXTrackHitsLocalX});
  }

  saveHist(c, cOption, hexHists);
  saveHist(c, cOption, heyHists);
  saveHist(c, cOption, hexheyHists);

  saveHist(c, cOption, loohexHists);
  saveHist(c, cOption, looheyHists);
  saveHist(c, cOption, loohexheyHists);

  saveHist(c, cOption, loohexmhexHists);
  saveHist(c, cOption, looheymheyHists);

  saveHist(c, cOption, loohexthlxHists);

  // Residuals
  HistPainterConfig cfgResXY;
  cfgResXY.canvasXSize = 10000;
  cfgResXY.canvasYSize = 10000. / chipIds.size();
  cfgResXY.stats = true;
  cfgResXY.xTitle = "x^{hit} - x^{est}, [mm]";
  cfgResXY.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgResXY.xAdjustDivisions = true;
  cfgResXY.option = "col";

  HistPainterConfig cfgResX;
  cfgResX.canvasXSize = 10000;
  cfgResX.canvasYSize = 10000. / 3;
  cfgResX.stats = true;
  cfgResX.xTitle = "x^{hit} - x^{est}, [mm]";
  cfgResX.yTitle = "Prob.";
  cfgResX.xAdjustDivisions = true;
  cfgResX.xNDivisions = 5;

  HistPainterConfig cfgResY;
  cfgResY.canvasXSize = 10000;
  cfgResY.canvasYSize = 10000. / 3;
  cfgResY.stats = true;
  cfgResY.xTitle = "y^{hit} - y^{est}, [mm]";
  cfgResY.yTitle = "Prob.";
  cfgResY.xAdjustDivisions = true;
  cfgResY.xNDivisions = 5;

  saveHist(c, cOption,
           {{predictedResidualsXCumulativeHist, cfgResX},
            {filteredResidualsXCumulativeHist, cfgResX},
            {smoothedResidualsXCumulativeHist, cfgResX}});
  saveHist(c, cOption,
           {{predictedResidualsYCumulativeHist, cfgResY},
            {filteredResidualsYCumulativeHist, cfgResY},
            {smoothedResidualsYCumulativeHist, cfgResY}});

  std::vector<std::pair<TH1 &, HistPainterConfig>> frxyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> srxyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> loorxyHists;

  std::vector<std::pair<TH1 &, HistPainterConfig>> prxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> frxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> srxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> loorxHists;

  std::vector<std::pair<TH1 &, HistPainterConfig>> pryHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> fryHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> sryHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> looryHists;
  cfgResX.canvasYSize = 10000. / chipIds.size();
  cfgResY.canvasYSize = 10000. / chipIds.size();
  for (std::size_t id : chipIds) {
    frxyHists.push_back({filteredResidualsXYHists.at(id), cfgResXY});
    srxyHists.push_back({smoothedResidualsXYHists.at(id), cfgResXY});
    loorxyHists.push_back({leaveOneOutResidualsXYHists.at(id), cfgResXY});

    prxHists.push_back({predictedResidualsXHists.at(id), cfgResX});
    frxHists.push_back({filteredResidualsXHists.at(id), cfgResX});
    srxHists.push_back({smoothedResidualsXHists.at(id), cfgResX});
    loorxHists.push_back({leaveOneOutResidualsXHists.at(id), cfgResX});

    pryHists.push_back({predictedResidualsYHists.at(id), cfgResY});
    fryHists.push_back({filteredResidualsYHists.at(id), cfgResY});
    sryHists.push_back({smoothedResidualsYHists.at(id), cfgResY});
    looryHists.push_back({leaveOneOutResidualsYHists.at(id), cfgResY});
  }
  saveHist(c, cOption, frxyHists);
  saveHist(c, cOption, srxyHists);
  saveHist(c, cOption, loorxyHists);

  saveHist(c, cOption, prxHists);
  saveHist(c, cOption, frxHists);
  saveHist(c, cOption, srxHists);
  saveHist(c, cOption, loorxHists);

  saveHist(c, cOption, pryHists);
  saveHist(c, cOption, fryHists);
  saveHist(c, cOption, sryHists);
  saveHist(c, cOption, looryHists);

  HistPainterConfig cfgResXResX;
  cfgResXResX.canvasXSize = 10000;
  cfgResXResX.canvasYSize = 10000. / (chipIds.size() - 1);
  cfgResXResX.stats = true;
  cfgResXResX.xTitle = "x^{hit} - x^{est}, [mm]";
  cfgResXResX.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgResXResX.xAdjustDivisions = true;
  cfgResXResX.xNDivisions = 5;

  HistPainterConfig cfgResYResY;
  cfgResYResY.canvasXSize = 10000;
  cfgResYResY.canvasYSize = 10000. / (chipIds.size() - 1);
  cfgResYResY.stats = true;
  cfgResYResY.xTitle = "y^{hit} - y^{est}, [mm]";
  cfgResYResY.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgResYResY.xAdjustDivisions = true;
  cfgResYResY.xNDivisions = 5;

  std::sort(chipIds_.begin(), chipIds_.end());
  for (std::size_t n = 0; n < chipIds_.size() - 1; n++) {
    std::size_t idN = chipIds_.at(n);
    std::vector<std::pair<TH1 &, HistPainterConfig>> resxresxHists;
    for (std::size_t m = n + 1; m < chipIds_.size(); m++) {
      std::size_t idM = chipIds_.at(m);
      resxresxHists.push_back(
          {smoothedResidualsXXHists.at(idN * 100 + idM), cfgResXResX});
    }
    for (std::size_t m = resxresxHists.size(); m < chipIds_.size() - 1; m++) {
      resxresxHists.push_back({dummy, cfgResXResX});
    }
    saveHist(c, cOption, resxresxHists);
  }
  for (std::size_t n = 0; n < chipIds_.size() - 1; n++) {
    std::size_t idN = chipIds_.at(n);
    std::vector<std::pair<TH1 &, HistPainterConfig>> resyresyHists;
    for (std::size_t m = n + 1; m < chipIds_.size(); m++) {
      std::size_t idM = chipIds_.at(m);
      resyresyHists.push_back(
          {smoothedResidualsYYHists.at(idN * 100 + idM), cfgResYResY});
    }
    for (std::size_t m = resyresyHists.size(); m < chipIds_.size() - 1; m++) {
      resyresyHists.push_back({dummy, cfgResYResY});
    }
    saveHist(c, cOption, resyresyHists);
  }

  // Pulls
  HistPainterConfig cfgPullX;
  cfgPullX.canvasXSize = 10000;
  cfgPullX.canvasYSize = 10000. / 3;
  cfgPullX.stats = true;
  cfgPullX.xTitle = "(x^{hit} - x^{est})/#sigma^{est}_{x}";
  cfgPullX.yTitle = "Prob.";
  cfgPullX.stats = true;
  cfgPullX.xAdjustDivisions = true;
  cfgPullX.xNDivisions = 5;

  HistPainterConfig cfgPullY;
  cfgPullY.canvasXSize = 10000;
  cfgPullY.canvasYSize = 10000. / 3;
  cfgPullY.stats = true;
  cfgPullY.xTitle = "(y^{hit} - y^{est})/#sigma^{est}_{y}";
  cfgPullY.yTitle = "Prob.";
  cfgPullY.stats = true;
  cfgPullY.xAdjustDivisions = true;
  cfgPullY.xNDivisions = 5;

  saveHist(c, cOption,
           {{predictedPullsXCumulativeHist, cfgPullX},
            {filteredPullsXCumulativeHist, cfgPullX},
            {smoothedPullsXCumulativeHist, cfgPullX}});
  saveHist(c, cOption,
           {{predictedPullsYCumulativeHist, cfgPullY},
            {filteredPullsYCumulativeHist, cfgPullY},
            {smoothedPullsYCumulativeHist, cfgPullY}});

  std::vector<std::pair<TH1 &, HistPainterConfig>> ppxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> fpxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> spxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> loopxHists;

  std::vector<std::pair<TH1 &, HistPainterConfig>> ppyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> fpyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> spyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> loopyHists;
  cfgPullX.canvasYSize = 10000. / chipIds_.size();
  cfgPullY.canvasYSize = 10000. / chipIds_.size();
  for (std::size_t id : chipIds_) {
    ppxHists.push_back({predictedPullsXHists.at(id), cfgPullX});
    fpxHists.push_back({filteredPullsXHists.at(id), cfgPullX});
    spxHists.push_back({smoothedPullsXHists.at(id), cfgPullX});
    loopxHists.push_back({leaveOneOutPullsXHists.at(id), cfgPullX});

    ppyHists.push_back({predictedPullsYHists.at(id), cfgPullY});
    fpyHists.push_back({filteredPullsYHists.at(id), cfgPullY});
    spyHists.push_back({smoothedPullsYHists.at(id), cfgPullY});
    loopyHists.push_back({leaveOneOutPullsYHists.at(id), cfgPullX});
  }
  saveHist(c, cOption, ppxHists);
  saveHist(c, cOption, fpxHists);
  saveHist(c, cOption, spxHists);
  saveHist(c, cOption, loopxHists);

  saveHist(c, cOption, ppyHists);
  saveHist(c, cOption, fpyHists);
  saveHist(c, cOption, spyHists);
  saveHist(c, cOption, loopyHists);

  // Momentum
  HistPainterConfig cfgPx;
  cfgPx.canvasXSize = 10000;
  cfgPx.canvasYSize = 10000. / 3;
  cfgPx.stats = true;
  cfgPx.xTitle = "P^{est}_{x}, [GeV]";
  cfgPx.yTitle = "Tracks/BX";
  cfgPx.stats = true;
  cfgPx.xAdjustDivisions = true;
  cfgPx.xNDivisions = 5;

  HistPainterConfig cfgPy;
  cfgPy.canvasXSize = 10000;
  cfgPy.canvasYSize = 10000. / 3;
  cfgPy.stats = true;
  cfgPy.xTitle = "P^{est}_{y}, [GeV]";
  cfgPy.yTitle = "Tracks/BX";
  cfgPy.stats = true;
  cfgPy.xAdjustDivisions = true;
  cfgPy.xNDivisions = 5;

  HistPainterConfig cfgPz;
  cfgPz.canvasXSize = 10000;
  cfgPz.canvasYSize = 10000. / 3;
  cfgPz.stats = true;
  cfgPz.xTitle = "P^{est}_{z}, [GeV]";
  cfgPz.yTitle = "Tracks/BX";
  cfgPz.stats = true;
  cfgPz.xAdjustDivisions = true;
  cfgPz.xNDivisions = 5;

  saveHist(c, cOption,
           {{originMomentumEstPxHist, cfgPx},
            {originMomentumEstPyHist, cfgPy},
            {originMomentumEstPzHist, cfgPz}});

  // Momentum direction
  HistPainterConfig cfgDx;
  cfgDx.canvasXSize = 10000;
  cfgDx.canvasYSize = 10000. / 3;
  cfgDx.stats = true;
  cfgDx.xTitle = "D^{est}_{x}";
  cfgDx.yTitle = "Tracks/BX";
  cfgDx.stats = true;
  cfgDx.xAdjustDivisions = true;
  cfgDx.xNDivisions = 5;

  HistPainterConfig cfgDy;
  cfgDy.canvasXSize = 10000;
  cfgDy.canvasYSize = 10000. / 3;
  cfgDy.stats = true;
  cfgDy.xTitle = "D^{est}_{y}";
  cfgDy.yTitle = "Tracks/BX";
  cfgDy.stats = true;
  cfgDy.xAdjustDivisions = true;
  cfgDy.xNDivisions = 5;

  HistPainterConfig cfgDz;
  cfgDz.canvasXSize = 10000;
  cfgDz.canvasYSize = 10000. / 3;
  cfgDz.stats = true;
  cfgDz.xTitle = "D^{est}_{z}";
  cfgDz.yTitle = "Tracks/BX";
  cfgDz.stats = true;
  cfgDz.xAdjustDivisions = true;
  cfgDz.xNDivisions = 5;

  saveHist(c, cOption,
           {{originMomentumEstDxHist, cfgDx},
            {originMomentumEstDyHist, cfgDy},
            {originMomentumEstDzHist, cfgDz}});

  HistPainterConfig cfgMomPhi;
  cfgMomPhi.canvasXSize = 10000;
  cfgMomPhi.canvasYSize = 10000. / 2;
  cfgMomPhi.stats = true;
  cfgMomPhi.xTitle = "P^{est}_{#phi}, [rad]";
  cfgMomPhi.yTitle = "Tracks/BX";
  cfgMomPhi.stats = true;
  cfgMomPhi.xAdjustDivisions = true;
  cfgMomPhi.xNDivisions = 5;

  HistPainterConfig cfgMomTheta;
  cfgMomTheta.canvasXSize = 10000;
  cfgMomTheta.canvasYSize = 10000. / 2;
  cfgMomTheta.stats = true;
  cfgMomTheta.xTitle = "P^{est}_{#theta}, [rad]";
  cfgMomTheta.yTitle = "Tracks/BX";
  cfgMomTheta.stats = true;
  cfgMomTheta.xAdjustDivisions = true;
  cfgMomTheta.xNDivisions = 5;

  saveHist(c, cOption,
           {{originMomentumEstPhiHist, cfgMomPhi},
            {originMomentumEstThetaHist, cfgMomTheta}});

  // Vertex
  HistPainterConfig cfgVxVy;
  cfgVxVy.stats = true;
  cfgVxVy.xTitle = "v_{x}, [mm]";
  cfgVxVy.yTitle = "v_{y}, [mm]";
  cfgVxVy.option = "col";
  saveHist(c, cOption, {vertexEstXYHist, cfgVxVy});

  HistPainterConfig cfgVx;
  cfgVx.canvasXSize = 10000;
  cfgVx.canvasYSize = 10000. / 2;
  cfgVx.stats = true;
  cfgVx.xTitle = "v_{x}, [mm]";
  cfgVx.yTitle = "Tracks/BX";
  cfgVx.stats = true;
  cfgVx.xAdjustDivisions = true;
  cfgVx.xNDivisions = 5;

  HistPainterConfig cfgVy;
  cfgVy.canvasXSize = 10000;
  cfgVy.canvasYSize = 10000. / 2;
  cfgVy.stats = true;
  cfgVy.xTitle = "v_{y}, [mm]";
  cfgVy.yTitle = "Tracks/BX";
  cfgVy.stats = true;
  cfgVy.xAdjustDivisions = true;
  cfgVy.xNDivisions = 5;

  saveHist(c, cOption, {{vertexEstXHist, cfgVx}, {vertexEstYHist, cfgVy}});

  // Track residuals vs track hit parameters
  HistPainterConfig cfgRxHx;
  cfgRxHx.canvasXSize = 10000;
  cfgRxHx.canvasYSize = 10000. / chipIds_.size();
  cfgRxHx.stats = false;
  cfgRxHx.xTitle = "x^{hit}, [mm]";
  cfgRxHx.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgRxHx.option = "col";

  HistPainterConfig cfgRxHy;
  cfgRxHy.canvasXSize = 10000;
  cfgRxHy.canvasYSize = 10000. / chipIds_.size();
  cfgRxHy.xTitle = "y^{hit}, [mm]";
  cfgRxHy.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgRxHy.option = "col";

  HistPainterConfig cfgRxHphi;
  cfgRxHphi.canvasXSize = 10000;
  cfgRxHphi.canvasYSize = 10000. / chipIds_.size();
  cfgRxHphi.xTitle = "#phi^{hit}, [rad]";
  cfgRxHphi.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgRxHphi.option = "col";

  HistPainterConfig cfgRyHx;
  cfgRyHx.canvasXSize = 10000;
  cfgRyHx.canvasYSize = 10000. / chipIds_.size();
  cfgRyHx.xTitle = "x^{hit}, [mm]";
  cfgRyHx.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgRyHx.option = "col";

  HistPainterConfig cfgRyHy;
  cfgRyHy.canvasXSize = 10000;
  cfgRyHy.canvasYSize = 10000. / chipIds_.size();
  cfgRyHy.xTitle = "y^{hit}, [mm]";
  cfgRyHy.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgRyHy.option = "col";

  HistPainterConfig cfgRyHphi;
  cfgRyHphi.canvasXSize = 10000;
  cfgRyHphi.canvasYSize = 10000. / chipIds_.size();
  cfgRyHphi.xTitle = "#phi^{hit}, [rad]";
  cfgRyHphi.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgRyHphi.option = "col";

  HistPainterConfig cfgRxHex;
  cfgRxHex.canvasXSize = 10000;
  cfgRxHex.canvasYSize = 10000. / chipIds_.size();
  cfgRxHex.xTitle = "#sigma_{x_{hit}}, [mm]";
  cfgRxHex.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgRxHex.option = "col";

  HistPainterConfig cfgRyHey;
  cfgRyHey.canvasXSize = 10000;
  cfgRyHey.canvasYSize = 10000. / chipIds_.size();
  cfgRyHey.xTitle = "#sigma_{y_{hit}}, [mm]";
  cfgRyHey.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgRyHey.option = "col";

  std::vector<std::pair<TH1 &, HistPainterConfig>> rxhxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> rxhyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> rxhphiHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> ryhxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> ryhyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> ryhphiHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> rxhexHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> ryheyHists;
  for (std::size_t id : chipIds_) {
    rxhxHists.push_back(
        {smoothedResidualsXTrackHitsLocalXHists.at(id), cfgRxHx});
    rxhyHists.push_back(
        {smoothedResidualsXTrackHitsLocalYHists.at(id), cfgRxHy});
    rxhphiHists.push_back(
        {smoothedResidualsXTrackHitsLocalPhiHists.at(id), cfgRxHphi});
    ryhxHists.push_back(
        {smoothedResidualsYTrackHitsLocalXHists.at(id), cfgRyHx});
    ryhyHists.push_back(
        {smoothedResidualsYTrackHitsLocalYHists.at(id), cfgRyHy});
    ryhphiHists.push_back(
        {smoothedResidualsYTrackHitsLocalPhiHists.at(id), cfgRyHphi});
    rxhexHists.push_back(
        {smoothedResidualsXTrackHitsErrorXHists.at(id), cfgRxHex});
    ryheyHists.push_back(
        {smoothedResidualsYTrackHitsErrorYHists.at(id), cfgRyHey});
  }
  saveHist(c, cOption, rxhxHists);
  saveHist(c, cOption, rxhyHists);
  saveHist(c, cOption, rxhphiHists);
  saveHist(c, cOption, ryhxHists);
  saveHist(c, cOption, ryhyHists);
  saveHist(c, cOption, ryhphiHists);
  saveHist(c, cOption, rxhexHists);
  saveHist(c, cOption, ryheyHists);

  // Track residuals vs track parameters
  HistPainterConfig cfgRxVx;
  cfgRxVx.canvasXSize = 10000;
  cfgRxVx.canvasYSize = 10000. / chipIds_.size();
  cfgRxVx.xTitle = "v^{est}_{x}, [mm]";
  cfgRxVx.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgRxVx.option = "col";

  HistPainterConfig cfgRxVy;
  cfgRxVy.canvasXSize = 10000;
  cfgRxVy.canvasYSize = 10000. / chipIds_.size();
  cfgRxVy.xTitle = "v^{est}_{y}, [mm]";
  cfgRxVy.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgRxVy.option = "col";

  HistPainterConfig cfgRyVx;
  cfgRyVx.canvasXSize = 10000;
  cfgRyVx.canvasYSize = 10000. / chipIds_.size();
  cfgRyVx.xTitle = "v^{est}_{x}, [mm]";
  cfgRyVx.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgRyVx.option = "col";

  HistPainterConfig cfgRyVy;
  cfgRyVy.canvasXSize = 10000;
  cfgRyVy.canvasYSize = 10000. / chipIds_.size();
  cfgRyVy.xTitle = "v^{est}_{y}, [mm]";
  cfgRyVy.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgRyVy.option = "col";

  HistPainterConfig cfgRxPy;
  cfgRxPy.canvasXSize = 10000;
  cfgRxPy.canvasYSize = 10000. / chipIds_.size();
  cfgRxPy.xTitle = "P^{est}_{y}, [GeV]";
  cfgRxPy.yTitle = "x^{hit} - x^{est}, [mm]";
  cfgRxPy.option = "col";

  HistPainterConfig cfgRyPy;
  cfgRyPy.canvasXSize = 10000;
  cfgRyPy.canvasYSize = 10000. / chipIds_.size();
  cfgRyPy.xTitle = "P^{est}_{y}, [GeV]";
  cfgRyPy.yTitle = "y^{hit} - y^{est}, [mm]";
  cfgRyPy.option = "col";

  std::vector<std::pair<TH1 &, HistPainterConfig>> rxvxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> rxvyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> ryvxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> ryvyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> rxpyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> rypyHists;
  for (std::size_t id : chipIds_) {
    rxvxHists.push_back({smoothedResidualsXVertexEstXHists.at(id), cfgRxVx});
    rxvyHists.push_back({smoothedResidualsXVertexEstYHists.at(id), cfgRxVy});
    ryvxHists.push_back({smoothedResidualsYVertexEstXHists.at(id), cfgRyVx});
    ryvyHists.push_back({smoothedResidualsYVertexEstYHists.at(id), cfgRyVy});
    rxpyHists.push_back(
        {smoothedResidualsYOriginMomentumEstPyHists.at(id), cfgRxPy});
    rypyHists.push_back(
        {smoothedResidualsYOriginMomentumEstPyHists.at(id), cfgRyPy});
  }
  saveHist(c, cOption, rxvxHists);
  saveHist(c, cOption, rxvyHists);
  saveHist(c, cOption, ryvxHists);
  saveHist(c, cOption, ryvyHists);
  saveHist(c, cOption, rxpyHists);
  saveHist(c, cOption, rypyHists);

  // Track pulls vs track hit parameters
  HistPainterConfig cfgPxHx;
  cfgPxHx.canvasXSize = 10000;
  cfgPxHx.canvasYSize = 10000. / chipIds_.size();
  cfgPxHx.stats = false;
  cfgPxHx.xTitle = "x^{hit}, [mm]";
  cfgPxHx.yTitle = "(x^{hit} - x^{est})/#sigma^{est}_{x}";
  cfgPxHx.option = "col";

  HistPainterConfig cfgPxHy;
  cfgPxHy.canvasXSize = 10000;
  cfgPxHy.canvasYSize = 10000. / chipIds_.size();
  cfgPxHy.xTitle = "y^{hit}, [mm]";
  cfgPxHy.yTitle = "(x^{hit} - x^{est})/#sigma^{est}_{x}";
  cfgPxHy.option = "col";

  HistPainterConfig cfgPyHx;
  cfgPyHx.canvasXSize = 10000;
  cfgPyHx.canvasYSize = 10000. / chipIds_.size();
  cfgPyHx.xTitle = "x_{hit}, [mm]";
  cfgPyHx.yTitle = "(y^{hit} - y^{est})/#sigma^{est}_{y}";
  cfgPyHx.option = "col";

  HistPainterConfig cfgPyHy;
  cfgPyHy.canvasXSize = 10000;
  cfgPyHy.canvasYSize = 10000. / chipIds_.size();
  cfgPyHy.xTitle = "y^{hit}, [mm]";
  cfgPyHy.yTitle = "(y^{hit} - y^{est})/#sigma^{est}_{y}";
  cfgPyHy.option = "col";

  std::vector<std::pair<TH1 &, HistPainterConfig>> pxhxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> pxhyHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> pyhxHists;
  std::vector<std::pair<TH1 &, HistPainterConfig>> pyhyHists;
  for (std::size_t id : chipIds_) {
    pxhxHists.push_back({smoothedPullsXTrackHitsLocalXHists.at(id), cfgPxHx});
    pxhyHists.push_back({smoothedPullsXTrackHitsLocalYHists.at(id), cfgPxHy});
    pyhxHists.push_back({smoothedPullsYTrackHitsLocalXHists.at(id), cfgPyHx});
    pyhyHists.push_back({smoothedPullsYTrackHitsLocalYHists.at(id), cfgPyHy});
  }
  saveHist(c, cOption, pxhxHists);
  saveHist(c, cOption, pxhyHists);
  saveHist(c, cOption, pyhxHists);
  saveHist(c, cOption, pyhyHists);

  // Cluster shape
  HistPainterConfig cfgClusterMatrix;
  cfgClusterMatrix.canvasXSize = 10000;
  cfgClusterMatrix.canvasYSize = 10000 / 5.0;
  cfgClusterMatrix.xTitle = "x, [pix]";
  cfgClusterMatrix.yTitle = "y, [pix]";
  cfgClusterMatrix.option = "col";

  HistPainterConfig cfgClusterMatrixLooResX;
  cfgClusterMatrixLooResX.canvasXSize = 10000;
  cfgClusterMatrixLooResX.canvasYSize = 10000 / 5.0;
  cfgClusterMatrixLooResX.stats = true;
  cfgClusterMatrixLooResX.xTitle = "x_{hit} - x_{loo}, [mm]";
  cfgClusterMatrixLooResX.yTitle = "Prob.";
  cfgClusterMatrixLooResX.option = "col";

  HistPainterConfig cfgClusterMatrixLooResY;
  cfgClusterMatrixLooResY.canvasXSize = 10000;
  cfgClusterMatrixLooResY.canvasYSize = 10000 / 5.0;
  cfgClusterMatrixLooResY.stats = true;
  cfgClusterMatrixLooResY.xTitle = "y_{hit} - y_{loo}, [mm]";
  cfgClusterMatrixLooResY.yTitle = "Prob.";
  cfgClusterMatrixLooResY.option = "col";

  HistPainterConfig cfgClusterMatrixLooErrorX;
  cfgClusterMatrixLooErrorX.canvasXSize = 10000;
  cfgClusterMatrixLooErrorX.canvasYSize = 10000 / 5.0;
  cfgClusterMatrixLooErrorX.stats = true;
  cfgClusterMatrixLooErrorX.xTitle = "#sigma_{meas, x, loo}, [mm]";
  cfgClusterMatrixLooErrorX.yTitle = "Prob.";
  cfgClusterMatrixLooErrorX.option = "col";

  HistPainterConfig cfgClusterMatrixLooErrorY;
  cfgClusterMatrixLooErrorY.canvasXSize = 10000;
  cfgClusterMatrixLooErrorY.canvasYSize = 10000 / 5.0;
  cfgClusterMatrixLooErrorY.stats = true;
  cfgClusterMatrixLooErrorY.xTitle = "#sigma_{meas, y, loo}, [mm]";
  cfgClusterMatrixLooErrorY.yTitle = "Prob.";
  cfgClusterMatrixLooErrorY.option = "col";

  // Dummy
  HistPainterConfig dummyCfg;
  saveHist(c, cOptionLast, {dummy, dummyCfg});
}
