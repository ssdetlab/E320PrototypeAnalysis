#pragma once

#include <cstddef>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "DetectorEvent.hpp"
#include "TChain.h"
#include "TF1.h"
#include "TFile.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TTree.h"
#include "TVector3.h"
#include "TVectorD.h"

class RootTrackAnalyzer {
 public:
  /// @brief diagnostic plots struct
  class Stats {
   public:
    struct Options {
      /// Chip ids
      std::set<std::size_t> chipIds;

      /// Map of Acts chip IDs to EUDAQ chip IDs
      std::map<std::size_t, std::size_t> chipIdsMap;

      /// X count
      std::size_t xCountBins;
      double xCountMin;
      double xCountMax;

      /// Mag conf hist parameters
      std::size_t dipoleStrengthBins;
      double dipoleStrengthMin;
      double dipoleStrengthMax;

      std::size_t xCorrectorStrengthBins;
      double xCorrectorStrengthMin;
      double xCorrectorStrengthMax;

      /// Track multiplicity hist parameters
      std::size_t nTracksBins;
      double nTracksMin;
      double nTracksMax;

      /// Cluster multiplicity hist parameters
      std::size_t nClustersBins;
      double nClustersMin;
      double nClustersMax;

      /// Cluster size hist parameters
      std::size_t clusterSizeBins;
      double clusterSizeMin;
      double clusterSizeMax;

      std::size_t clusterDXBins;
      double clusterDXMin;
      double clusterDXMax;

      std::size_t clusteDYrBins;
      double clusterDYSizeMin;
      double clusterDYSizeMax;

      /// Chi2 hists parameters
      std::size_t chi2Bins;
      double chi2Min;
      double chi2Max;

      /// Local track hits hists parameters
      std::size_t trackHitsLocalBinsX;
      std::size_t trackHitsLocalBinsY;
      double trackHitsLocalMinX;
      double trackHitsLocalMaxX;
      double trackHitsLocalMinY;
      double trackHitsLocalMaxY;

      /// Track hits errors parameters
      std::size_t trackHitsErrorBinsX;
      std::size_t trackHitsErrorBinsY;
      double trackHitsErrorMinX;
      double trackHitsErrorMaxX;
      double trackHitsErrorMinY;
      double trackHitsErrorMaxY;

      // Track hits leave-one-out variance hists parameters
      std::size_t trackHitsLooErrorBinsX;
      std::size_t trackHitsLooErrorBinsY;
      double trackHitsLooErrorMinX;
      double trackHitsLooErrorMaxX;
      double trackHitsLooErrorMinY;
      double trackHitsLooErrorMaxY;

      /// Resiudals hists parameters
      std::size_t residualsXBins;
      double residualsXMin;
      double residualsXMax;

      std::size_t residualsYBins;
      double residualsYMin;
      double residualsYMax;

      /// Leave-one-out resiudals hists parameters
      std::size_t looResidualsXBins;
      double looResidualsXMin;
      double looResidualsXMax;

      std::size_t looResidualsYBins;
      double looResidualsYMin;
      double looResidualsYMax;

      /// Pulls hists parameters
      std::size_t pullsXBins;
      double pullsXMin;
      double pullsXMax;

      std::size_t pullsYBins;
      double pullsYMin;
      double pullsYMax;

      /// Momentum hists parameters
      std::size_t momentumXBins;
      double momentumXMin;
      double momentumXMax;

      std::size_t momentumYBins;
      double momentumYMin;
      double momentumYMax;

      std::size_t momentumZBins;
      double momentumZMin;
      double momentumZMax;

      /// Momentum direction hists parameters
      std::size_t momentumDirXBins;
      double momentumDirXMin;
      double momentumDirXMax;

      std::size_t momentumDirYBins;
      double momentumDirYMin;
      double momentumDirYMax;

      std::size_t momentumDirZBins;
      double momentumDirZMin;
      double momentumDirZMax;

      std::size_t momentumPhiBins;
      double momentumPhiMin;
      double momentumPhiMax;

      std::size_t momentumThetaBins;
      double momentumThetaMin;
      double momentumThetaMax;

      /// Vertex hists parameters
      std::size_t vertexXBins;
      double vertexXMin;
      double vertexXMax;

      std::size_t vertexYBins;
      double vertexYMin;
      double vertexYMax;
    };

    Stats() = default;

    Options opt;

    void init(const std::string& pfix, const Options& opt);

    void reset();

    void print();

    std::string prefix = "";
    std::set<std::size_t> chipIds;

    /// Counters and cut flow
    std::size_t nTotal = 0;
    std::size_t nPassedOccupancy = 0;
    std::size_t nPassedParity = 0;
    std::size_t nPassedXCount = 0;
    std::size_t nPassedChi2Predicted = 0;
    std::size_t nPassedChi2Filtered = 0;
    std::size_t nPassedChi2Smoothed = 0;
    std::size_t nPassedVertexZ = 0;
    std::size_t nPassedVertexY = 0;
    std::size_t nPassedMomentumZ = 0;
    std::size_t nPassedResiduals = 0;

    /// Cut flow
    TH1D cutFlowHist;

    /// X count
    TH1D xCountHist;

    /// Mag conf hist
    TH2D magnetConfHist;

    /// Number of tracks/BX vs number of clusters/BX
    std::map<std::size_t, TH1D> nClustersHist;
    TH1D nTracksHist;
    TH2D nTracksNClustersHist;

    /// Cluster lengths hists
    TH2D clusterLxLyHist;
    TH1D clusterLxHist;
    TH1D clusterLyHist;

    /// Chi2
    TH1D chi2PredictedHist;
    TH1D chi2FilteredHist;
    TH1D chi2SmoothedHist;

    /// Chi2 functions
    TH2D chi2SmoothedXCountHist;

    /// Local track hits
    std::map<std::size_t, TH2D> trackHitsLocalHists;

    std::map<std::size_t, TH2D> trackHitsLocalXXHists;
    std::map<std::size_t, TH2D> trackHitsLocalYYHists;

    /// Track hits errors
    std::map<std::size_t, TH1D> trackHitsErrorXHists;
    std::map<std::size_t, TH1D> trackHitsErrorYHists;
    std::map<std::size_t, TH2D> trackHitsErrorXYHists;

    std::map<std::size_t, TH1D> leaveOneOutTrackHitsErrorXHists;
    std::map<std::size_t, TH1D> leaveOneOutTrackHitsErrorYHists;
    std::map<std::size_t, TH2D> leaveOneOutTrackHitsErrorXYHists;

    std::map<std::size_t, TH2D> looTrackHitsErrorModelTrackHitsErrorXHists;
    std::map<std::size_t, TH2D> looTrackHitsErrorModelTrackHitsErrorYHists;

    std::map<std::size_t, TH2D> looTrackHitsErrorXTrackHitsLocalXHists;

    /// Residuals
    TH1D predictedResidualsXCumulativeHist;
    TH1D filteredResidualsXCumulativeHist;
    TH1D smoothedResidualsXCumulativeHist;

    TH1D predictedResidualsYCumulativeHist;
    TH1D filteredResidualsYCumulativeHist;
    TH1D smoothedResidualsYCumulativeHist;

    std::map<std::size_t, TH2D> filteredResidualsXYHists;
    std::map<std::size_t, TH2D> smoothedResidualsXYHists;
    std::map<std::size_t, TH2D> leaveOneOutResidualsXYHists;

    std::map<std::size_t, TH2D> smoothedResidualsXXHists;
    std::map<std::size_t, TH2D> smoothedResidualsYYHists;

    std::map<std::size_t, TH1D> predictedResidualsXHists;
    std::map<std::size_t, TH1D> filteredResidualsXHists;
    std::map<std::size_t, TH1D> smoothedResidualsXHists;
    std::map<std::size_t, TH1D> leaveOneOutResidualsXHists;

    std::map<std::size_t, TH1D> predictedResidualsYHists;
    std::map<std::size_t, TH1D> filteredResidualsYHists;
    std::map<std::size_t, TH1D> smoothedResidualsYHists;
    std::map<std::size_t, TH1D> leaveOneOutResidualsYHists;

    /// Pulls
    TH1D predictedPullsXCumulativeHist;
    TH1D filteredPullsXCumulativeHist;
    TH1D smoothedPullsXCumulativeHist;

    TH1D predictedPullsYCumulativeHist;
    TH1D filteredPullsYCumulativeHist;
    TH1D smoothedPullsYCumulativeHist;

    std::map<std::size_t, TH1D> predictedPullsXHists;
    std::map<std::size_t, TH1D> filteredPullsXHists;
    std::map<std::size_t, TH1D> smoothedPullsXHists;
    std::map<std::size_t, TH1D> leaveOneOutPullsXHists;

    std::map<std::size_t, TH1D> predictedPullsYHists;
    std::map<std::size_t, TH1D> filteredPullsYHists;
    std::map<std::size_t, TH1D> smoothedPullsYHists;
    std::map<std::size_t, TH1D> leaveOneOutPullsYHists;

    /// Momentum
    TH1D originMomentumEstPxHist;
    TH1D originMomentumEstPyHist;
    TH1D originMomentumEstPzHist;

    /// Momentum direction
    TH1D originMomentumEstDxHist;
    TH1D originMomentumEstDyHist;
    TH1D originMomentumEstDzHist;

    TH1D originMomentumEstPhiHist;
    TH1D originMomentumEstThetaHist;

    /// Vertex
    TH2D vertexEstXYHist;
    TH1D vertexEstXHist;
    TH1D vertexEstYHist;

    /// Track residuals vs track hit parameters
    std::map<std::size_t, TH2D> smoothedResidualsXTrackHitsLocalXHists;
    std::map<std::size_t, TH2D> smoothedResidualsXTrackHitsLocalYHists;
    std::map<std::size_t, TH2D> smoothedResidualsXTrackHitsLocalPhiHists;
    std::map<std::size_t, TH2D> smoothedResidualsYTrackHitsLocalXHists;
    std::map<std::size_t, TH2D> smoothedResidualsYTrackHitsLocalYHists;
    std::map<std::size_t, TH2D> smoothedResidualsYTrackHitsLocalPhiHists;
    std::map<std::size_t, TH2D> smoothedResidualsXTrackHitsErrorXHists;
    std::map<std::size_t, TH2D> smoothedResidualsYTrackHitsErrorYHists;

    /// Track residuals vs track parameters
    std::map<std::size_t, TH2D> smoothedResidualsXVertexEstXHists;
    std::map<std::size_t, TH2D> smoothedResidualsXVertexEstYHists;
    std::map<std::size_t, TH2D> smoothedResidualsYVertexEstXHists;
    std::map<std::size_t, TH2D> smoothedResidualsYVertexEstYHists;

    std::map<std::size_t, TH2D> smoothedResidualsXOriginMomentumEstPyHists;
    std::map<std::size_t, TH2D> smoothedResidualsYOriginMomentumEstPyHists;

    /// Track pulls vs track hit parameters
    std::map<std::size_t, TH2D> smoothedPullsXTrackHitsLocalXHists;
    std::map<std::size_t, TH2D> smoothedPullsXTrackHitsLocalYHists;
    std::map<std::size_t, TH2D> smoothedPullsYTrackHitsLocalXHists;
    std::map<std::size_t, TH2D> smoothedPullsYTrackHitsLocalYHists;

    /// Cluster matrices
    std::map<std::size_t, TH2I> clusterMatricesHists;

    /// Cluster-matrix-specific leave-one-out distributions
    std::map<std::size_t, TH1D> clusterMatrixLooResXHists;
    std::map<std::size_t, TH1D> clusterMatrixLooResYHists;

    std::unordered_map<std::size_t, TH1D> clusterMatrixLooErrorXHists;
    std::unordered_map<std::size_t, TH1D> clusterMatrixLooErrorYHists;
  };

  /// @brief nested configuration struct
  struct Config {
    /// Chip ids
    std::set<std::size_t> chipIds;
    /// Input track tree name
    std::string trackTreeName;
    /// Path to input track files directory
    std::string inTrackDir;
    /// Input cluster tree name
    std::string clusterTreeName;
    /// Path to input cluster files directory
    std::string inClusterDir;
  };

  /// @brief options struct
  struct Options {
    /// Prefix
    std::string prefix;
    /// Stats options
    Stats::Options statsOpt;
    /// Number of BXs
    std::size_t nBXs;
    /// Cluster occupancy range
    std::size_t minOcc;
    std::size_t maxOcc;
    /// Flag to apply epics parity cut
    bool requireEpicsParity;
    /// Epics parity cut
    std::size_t epicsParity;
    /// Range of HT cell intersections counts
    std::size_t minXCount;
    std::size_t maxXCount;
    /// Chi2 predicted cut
    double minChi2Predicted = 0;
    double maxChi2Predicted = std::numeric_limits<double>::max();
    /// Chi2 filtered cut
    double minChi2Filtered = 0;
    double maxChi2Filtered = std::numeric_limits<double>::max();
    /// Chi2 smoothed cut
    double minChi2Smoothed = 0;
    double maxChi2Smoothed = std::numeric_limits<double>::max();
    /// Vertex Z cut
    double minVertexZ = -std::numeric_limits<double>::max();
    double maxVertexZ = std::numeric_limits<double>::max();
    /// Vertex Y cut
    double minVertexY = -std::numeric_limits<double>::max();
    double maxVertexY = std::numeric_limits<double>::max();
    /// Momentum X cut
    double minMomentumX = -std::numeric_limits<double>::max();
    double maxMomentumX = std::numeric_limits<double>::max();
    /// Slicing step width in number of bins
    std::size_t sliceStepSize = 20;
    /// Slicing step width in number of bins
    std::size_t sliceStepWidth = 20;
  };

  RootTrackAnalyzer(const RootTrackAnalyzer&) = delete;
  RootTrackAnalyzer(const RootTrackAnalyzer&&) = delete;

  /// @brief constructor
  explicit RootTrackAnalyzer(const Config& config);

  /// @brief write out data to the file
  void process(const Options& opt);

  /// Readonly access to the config
  const Config& config() const { return m_cfg; }

 private:
  /// Configuration
  Config m_cfg;

  /// Input track file
  TFile* m_trackFile = nullptr;

  /// Input track tree
  TTree* m_trackTree = nullptr;

  /// Track chain owner
  TChain* m_trackChainOwner = nullptr;

  /// Input cluster file
  TFile* m_clusterFile = nullptr;

  /// Input cluster tree
  TTree* m_clusterTree = nullptr;

  /// Cluster chain owner
  TChain* m_clusterChainOwner = nullptr;

 protected:
  ///--------------------------------------
  /// Cluster tree branches

  /// Event ID
  ULong64_t m_clusterEventId = 0;

  /// Detector event
  E320Io::DetectorEvent* m_detEv = nullptr;

  ///--------------------------------------
  /// Track tree branches

  /// EUDAQ trigger ID
  std::size_t m_eudaqTrgN = 0;

  /// EUDAQ DAQ run number
  std::size_t m_eudaqDAQNumber = 0;

  /// EUDAQ DAQ run start timestamp
  std::size_t m_eudaqRunStartTs = 0;

  /// EUDAQ DAQ run end timestamp
  std::size_t m_eudaqRunEndTs = 0;

  /// Event EPICS parity
  std::size_t m_epicsParity = 0;

  /// Event EPICS PID
  std::size_t m_epicsPulseId = 0;

  /// Event EPICS DAQ number
  std::size_t m_epicsDAQNumber = 0;

  /// Magnet configuration as seen by track
  double m_quad1Grad = 0;
  double m_quad2Grad = 0;
  double m_quad3Grad = 0;
  double m_xCorrectorStrength = 0;
  double m_dipoleStrength = 0;

  /// Measurement hits in the surface frame
  std::vector<TVector2>* m_trackHitsLocal = nullptr;

  /// Measurement hits in the global frame
  std::vector<TVector3>* m_trackHitsGlobal = nullptr;

  /// Direction measurements in the track frame
  std::vector<TVector3>* m_onSurfaceTrackDirection = nullptr;

  /// Covariances of the track hits
  std::vector<TMatrixD>* m_trackHitCovs = nullptr;

  /// Covariances of the track agnles
  std::vector<TMatrixD>* m_trackAngleCovs = nullptr;

  /// Measurements cluster matrices
  std::vector<TMatrixD>* m_trackHitClusterMatrices = nullptr;

  /// Measurements cluster matrices extents in local X
  std::vector<std::size_t>* m_trackHitClusterMatrixLengthsX = nullptr;

  /// Measurements cluster matrices extents in local Y
  std::vector<std::size_t>* m_trackHitClusterMatrixLengthsY = nullptr;

  /// Measurements cluster matrices pixel count
  std::vector<std::size_t>* m_trackHitClusterMatrixSizes = nullptr;

  /// Measurements cluster matrices shape IDs
  std::vector<std::size_t>* m_trackHitClusterMatrixShapeIds = nullptr;

  /// Measurements cluster matrices pixels
  std::vector<std::vector<std::pair<std::uint16_t, std::uint16_t>>>*
      m_trackHitClusterMatrixPixels = nullptr;

  /// Geometry ids of the track hits
  std::vector<std::size_t>* m_geometryIds = nullptr;

  /// KF predicted track hits in the surface frame
  std::vector<TVector2>* m_predictedTrackHitsLocal = nullptr;
  std::vector<TVector2>* m_filteredTrackHitsLocal = nullptr;
  std::vector<TVector2>* m_smoothedTrackHitsLocal = nullptr;

  std::vector<TVector2>* m_leaveOneOutTrackHitsLocal = nullptr;

  /// KF predicted track hits in the global frame
  std::vector<TVector3>* m_predictedTrackHitsGlobal = nullptr;
  std::vector<TVector3>* m_filteredTrackHitsGlobal = nullptr;
  std::vector<TVector3>* m_smoothedTrackHitsGlobal = nullptr;

  std::vector<TVector3>* m_leaveOneOutTrackHitsGlobal = nullptr;

  /// KF predicted on surface momenta in the track frame
  std::vector<TLorentzVector>* m_predictedOnSurfaceMomentum = nullptr;
  std::vector<TLorentzVector>* m_filteredOnSurfaceMomentum = nullptr;
  std::vector<TLorentzVector>* m_smoothedOnSurfaceMomentum = nullptr;

  /// KF residuals with respect to the measurements
  std::vector<TVector2>* m_predictedHitResiduals = nullptr;
  std::vector<TVector2>* m_filteredHitResiduals = nullptr;
  std::vector<TVector2>* m_smoothedHitResiduals = nullptr;

  std::vector<TVector2>* m_predictedAngleResiduals = nullptr;
  std::vector<TVector2>* m_filteredAngleResiduals = nullptr;
  std::vector<TVector2>* m_smoothedAngleResiduals = nullptr;

  std::vector<TVector2>* m_leaveOneOutHitResiduals = nullptr;

  /// KF pulls with respect to the measurements
  std::vector<TVector2>* m_predictedHitPulls = nullptr;
  std::vector<TVector2>* m_filteredHitPulls = nullptr;
  std::vector<TVector2>* m_smoothedHitPulls = nullptr;

  std::vector<TVector2>* m_predictedAnglePulls = nullptr;
  std::vector<TVector2>* m_filteredAnglePulls = nullptr;
  std::vector<TVector2>* m_smoothedAnglePulls = nullptr;

  std::vector<TVector2>* m_leaveOneOutHitPulls = nullptr;

  /// KF states covariances
  std::vector<TMatrixD>* m_predictedTrackStateCovs = nullptr;
  std::vector<TMatrixD>* m_filteredTrackStateCovs = nullptr;
  std::vector<TMatrixD>* m_smoothedTrackStateCovs = nullptr;
  std::vector<TMatrixD>* m_leaveOneOutTrackStateCovs = nullptr;

  /// Guessed bound track parameters
  TVectorD* m_boundTrackParametersGuess = nullptr;
  TMatrixD* m_boundTrackCovGuess = nullptr;

  /// KF predicted bound track parameters
  TVectorD* m_boundTrackParametersEst = nullptr;
  TMatrixD* m_boundTrackCovEst = nullptr;

  /// Initial guess of the momentum at the IP
  TLorentzVector* m_originMomentumGuess = nullptr;

  /// Initial guess of the vertex at the IP
  TVector3* m_vertexGuess = nullptr;

  /// KF predicted momentum at the IP
  TLorentzVector* m_originMomentumEst = nullptr;

  /// KF predicted vertex at the IP
  TVector3* m_vertexEst = nullptr;

  /// Chi2 of the track
  /// with respect ot the
  /// measurement
  double m_chi2Predicted = 0;
  double m_chi2Filtered = 0;
  double m_chi2Smoothed = 0;

  /// Number of degrees of freedom
  /// of the track
  std::size_t m_ndf = 0;

  /// Track ID
  std::size_t m_trackId = 0;

  /// Event ID
  std::size_t m_eventId = 0;

  /// PDG ID
  int m_pdgId = 0;

  /// Charge
  int m_charge = 0;

  /// HT cell intersections count
  std::size_t m_xCount = 0;
};
