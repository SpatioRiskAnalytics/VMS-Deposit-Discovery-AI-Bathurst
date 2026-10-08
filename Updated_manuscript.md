Camp-Scale Machine Learning Prospectivity Mapping of VMS Deposits in the Bathurst Mining Camp, New Brunswick: Integrating Geophysical Derivatives, Multi-Element Till Geochemistry, and Geologically Constrained Class Labels
Dele Falebita1 · Mohammad Parsa2 · David Lentz3
1 Tech Connect Southeast, Venn Innovation, Moncton, New Brunswick, Canada
2 Natural Resources Canada, Geological Survey of Canada, Ottawa, Ontario, Canada
3 Department of Earth Sciences, University of New Brunswick, Fredericton, New Brunswick, Canada
Corresponding author: Dele Falebita, dele@technb.ca; dele.1.falebita@gmail.com; ORCID: https://orcid.org/0000-0002-8154-5832

Acknowledgement
The authors appreciate the New Brunswick Department of Natural Resources for making the data available through their ArcGIS REST server.

Author Contributions
Dele Falebita performed the data analysis, interpreted the results, and prepared the original manuscript draft. Mohammad Parsa reviewed and edited the manuscript and provided technical guidance throughout the study. David Lentz provided supervision and oversight of the research.

Statements and Declarations
Conflict of Interest: The authors declare that they have no known financial or personal relationships that could have influenced the work reported in this paper.

Camp-Scale Machine Learning Prospectivity Mapping of VMS Deposits in the Bathurst Mining Camp, New Brunswick: Integrating Geophysical Derivatives, Multi-Element Till Geochemistry, and Geologically Constrained Class Labels

Abstract
We present a machine-learning framework for camp-scale volcanogenic massive sulphide (VMS) prospectivity mapping in the Bathurst Mining Camp (BMC), New Brunswick, Canada. Aeromagnetic, gravity, and radiometric datasets were integrated with a compiled 17-element till geochemistry dataset comprising 2,753 sample locations. Geophysical derivatives were generated to enhance structural features, while geochemical surfaces were interpolated using inverse distance weighting and transformed using centered log-ratio (CLR) methods. Principal Component Analysis (PCA), Factor Analysis (FA), and a geologically weighted Multi-Element Anomaly Score (MEAS) were used to extract geochemically meaningful predictors associated with VMS mineralization. Random Forest (RF) and Extreme Gradient Boosting (XGBoost) classifiers were trained using 295 spatial labels consisting of 45 known VMS deposits and 250 geologically constrained negative labels derived from barren drill intercepts and feature-space dissimilar samples. Model evaluation employed Synthetic Minority Over-sampling Technique (SMOTE) class balancing and 5-fold spatial cross-validation. RF outperformed XGBoost across the principal discrimination and targeting metrics, achieving Receiver Operating Characteristics-Area Under Curve (ROCAUC) values of 0.9318 +/- 0.0368 and 0.9098 +/- 0.0369, and Success Rate AUC values of 0.9680 and 0.9494 respectively. The RF model captured 91.1% of known VMS deposits within the highest-ranked 10% of the study area. Prospectivity maps produced by both classifiers delineated spatially coherent NNE-SSW-trending corridors that coincide with known VMS clusters and favourable Tetagouche Group volcanic horizons, while also identifying previously unrecognized target areas. The radiometric Thorium/Potassium (Th/K) alteration ratio emerged as the most influential predictor in both models, followed by molybdenum-, zinc-, and lead-related geochemical variables. The results demonstrate the value of integrating hydrothermal alteration signatures, structural geophysics, and multi-element till geochemistry for camp-scale VMS exploration targeting in covered terranes.

Highlights
Integrated regional geophysical and till-geochemical datasets were used to predict concealed VMS mineralization in the Bathurst Mining Camp using machine learning
Geologically Constrained Class Labels were used to improve the representation of non-mineralized conditions in machine learning training data.
Th/K emerged as the most influential predictor, highlighting the importance of hydrothermal alteration signatures for VMS prospectivity mapping.

Keywords: mineral prospectivity mapping; volcanogenic massive sulphide; Bathurst Mining Camp; till geochemistry; class negative label; geophysical derivatives

1. Introduction
Volcanogenic massive sulphide (VMS) deposits are major global repositories of base metals, including copper, zinc, and lead, as well as associated precious metals such as gold and silver. Several commodities commonly associated with VMS systems, including zinc, copper, indium, bismuth, tin, and antimony, are classified as critical minerals that are essential for green-energy transition technologies and global decarbonization efforts (Franklin et al., 2005; Galley et al., 2007). Historically, VMS discovery relied on identifying shallow, outcropping mineralization. However, most near-surface deposits within mature exploration districts have already been identified, forcing exploration programs to target concealed systems beneath glacial overburden and transported cover (Goodfellow and McCutcheon, 2003). In glaciated terrains, successful exploration increasingly depends on the integration of regional geophysical and geochemical datasets capable of detecting indirect signatures of mineralization.

The Bathurst Mining Camp (BMC) of northern New Brunswick, Canada, is one of the world's premier VMS districts, hosting more than 45 known deposits and representing a major centre of historical base-metal production (Goodfellow, 2007; Goodfellow & McCutcheon, 2003). In addition to its historical significance, the camp contains commodities such as zinc, copper, indium, bismuth, tin, and antimony that appear on Canada's 2024 Critical Minerals List (Natural Resources Canada, 2024). The camp is hosted within the Cambro-Ordovician Tetagouche Group and is characterized by structurally complex bimodal volcanic-sedimentary sequences that have undergone intense polyphase deformation and ductile shear-zone development (van Staal et al., 2003). This structural complexity, combined with variable thicknesses of glacial till, obscures the surface expression of mineralized zones and complicates exploration targeting. Despite decades of exploration and the availability of extensive geoscientific datasets, relatively few new discoveries have been made in recent years, while an estimated 70% of the camp's prospective ground remains untested beneath glacial cover or insufficiently explored (McCutcheon & Walker, 2020). This contrast between data abundance and exploration success highlights the need for improved approaches to regional-scale mineral prospectivity mapping.

Machine learning (ML) has emerged as a powerful framework for mineral prospectivity mapping (MPM) because it enables the integration of diverse geoscientific datasets and the identification of complex, non-linear relationships associated with mineralization (Carranza & Laborte, 2015; Rodriguez-Galiano et al., 2014; Zuo et al., 2019). Within the BMC, Parsa et al. (2023) demonstrated the effectiveness of ML-based prospectivity mapping for volcanogenic-hosted massive sulphide deposits using airborne magnetic, radiometric, electromagnetic, and till-geochemistry datasets from the EXTECH II program. However, that study was restricted to the northeastern Brunswick Belt and incorporated only three till-geochemistry elements (Pb, Zn, and Cu). Moreover, Parsa et al. (2023) reported that airborne electromagnetic data commonly yielded unreliable predictors because of the masking effects associated with conductive Carboniferous overburden. Consequently, no study has yet evaluated the predictive value of integrating the complete airborne geophysical compilation of Ugalde et al. (2018) with the full New Brunswick till-geochemistry database at the scale of the entire BMC. This represents a significant knowledge gap because the provincial till-geochemistry dataset captures a much broader spectrum of geological information, including hydrothermal pathfinder elements and critical-mineral indicators such as indium, bismuth, tin, and antimony. Whether camp-scale integration of these complementary datasets can improve the prediction of concealed VMS mineralization therefore remains an important unresolved question.

A second challenge concerns the effective integration and interpretation of geophysical and geochemical information in regional VMS exploration. These datasets capture fundamentally different but complementary aspects of the mineral system. Geophysical surveys image lithological boundaries, structures, and physical-property contrasts that influence hydrothermal fluid flow and ore deposition, whereas geochemical datasets record elemental dispersion patterns and alteration footprints associated with mineralizing processes. The effectiveness of ML-based prospectivity mapping therefore depends not only on the quantity of available data but also on the physical and statistical integrity of the predictor variables (Zuo et al., 2021). For geophysical datasets, structural information is commonly enhanced through derivative products that emphasize geological boundaries and potential fluid pathways. However, derivative calculations performed on reprojected or interpolated grids may introduce spatial artifacts and distort high-frequency geological signals (Thomas et al., 2000). Similarly, radiometric datasets provide valuable information on hydrothermal alteration through radioelement distributions and ratios associated with potassic and sericitic alteration around submarine vent systems (Shives et al., 1997).

Geochemical datasets introduce additional complexity because elemental concentrations are compositional in nature and are subject to the constant-sum constraint (Aitchison, 1986; Filzmoser et al., 2009). Under these conditions, conventional multivariate analyses can generate spurious relationships that reflect data closure rather than genuine geological processes. Compositional data analysis (CoDA) provides a rigorous framework for addressing these limitations (Egozcue et al., 2003). Nevertheless, the question of how best to reconcile broad multivariate geochemical signatures that characterize regional hydrothermal systems with localized elemental anomalies that may provide direct evidence of mineralization remains. This challenge is particularly important for till-geochemistry datasets, where regional trends and localized pathfinder signatures may contain complementary information about concealed VMS systems (Parkhill & Doiron, 2003).

A further challenge in data-driven MPM is the negative-label for mineral absence problem. Mineral prospectivity mapping is typically formulated as a binary classification task requiring both positive and negative training labels (Parsa & Cumani, 2025). While positive labels can be assigned to known mineral occurrences, identifying reliable negative labels is considerably more difficult because the absence of a known deposit does not guarantee the absence of mineralization (Carranza & Laborte, 2015). Consequently, negative-label uncertainty can strongly influence model performance and predictive outcomes. Recent studies have shown that geologically informed negative-label strategies can improve model discrimination and exploration targeting by reducing ambiguity within the negative class and providing a more defensible representation of negative evidence (Barbet-Massin et al., 2012; Maepa et al., 2021; Parsa & Cumani, 2025).

Spatial autocorrelation presents an additional obstacle to the development of reliable ML-based prospectivity models. Nearby observations commonly share similar geological, geophysical, and geochemical characteristics, resulting in statistical dependence within spatial datasets. When observations are partitioned randomly into training and validation subsets, model performance may be artificially inflated because validation samples are not truly independent of the training data (Brenning, 2012; Roberts et al., 2017). Obtaining realistic estimates of model performance therefore requires validation approaches that account for spatial dependence. Beyond predictive accuracy, exploration programs also require models that can efficiently prioritize targets for follow-up investigation. Consequently, measures of targeting efficiency are essential complements to conventional classification metrics and provide a more practical assessment of exploration value (Bonham-Carter, 1994; Carranza, 2008).

Against this background, we present a camp-scale machine-learning prospectivity analysis of the Bathurst Mining Camp, integrating the regional airborne geophysical compilation of Ugalde et al. (2018) with a unified 17-element New Brunswick till-geochemistry database. Specifically, this study addresses four key areas: (1) integration of regional geophysical and till geochemical datasets to support camp-scale prediction of concealed VMS mineralization; (2) identification of the geophysical and geochemical variables that contribute most strongly to prospectivity predictions; (3) application of geologically informed negative-label constraints to construct a more representative training dataset for machine learning-based prospectivity modelling; and (4) development of a data-driven framework for ranking and prioritizing exploration targets for follow-up investigation. By building on decades of geological, geochemical, and geophysical research in the Bathurst Mining Camp, we seek to enhance the targeting of concealed mineralization and advance mineral prospectivity mapping methodologies for mature, data-rich VMS districts elsewhere.

2. Regional Geological Setting

2.1 Tectonic and Stratigraphic Framework
The Bathurst Mining Camp (BMC) occupies the Gander Zone of the northern Appalachian Orogen in New Brunswick, Canada (Fig. 1; Rogers & van Staal, 2003; van Staal et al., 2003). Its geological architecture reflects the evolution of the Cambro-Ordovician Tetagouche-Four Falls back-arc basin, which formed during the rifting of the Popelogan arc from the Gondwanan passive margin. Subsequent Taconic, Salinic, and Acadian orogenic events progressively closed the basin and tectonically imbricated the volcanic and sedimentary successions into a series of thrust-bounded structural blocks (Goodfellow & McCutcheon, 2003; van Staal et al., 2003).

The geological map (Fig. 1) highlights the dominance of Middle Ordovician volcanic and sedimentary assemblages, particularly the Tetagouche Group, which occupies much of the central region and hosts most known mineral occurrences, including the Brunswick No. 12 (B12) and Brunswick No. 6 (B6) deposits. The Tetagouche Group comprises the Nepisiguit Falls Formation (felsic volcaniclastics and tuffs), the Flat Landing Brook Formation (rhyolite flows and hyaloclastites), and the Boucher Brook Formation (tholeiitic pillow basalts, black shales, and iron formation) (Goodfellow, 2007). The California Lake Group forms an extensive belt along the northern part of the camp and hosts additional VMS deposits, including Caribou and Restigouche.

The para-autochthonous Miramichi Group forms the basement succession and consists mainly of quartzarenites and carbonaceous argillites deposited on the Gondwanan passive margin prior to rifting. Other mapped units include the Fournier Group, Sheephouse Brook Group, Upsalquitch Gabbro, granitic intrusions, and younger Silurian-Carboniferous volcanic and sedimentary rocks. The distribution of lithological units and the abundance of thrust faults shown on Figure 1 emphasize the strong structural modification of the original basin architecture and the importance of tectonic juxtaposition in preserving and exposing mineralized stratigraphy.

2.2 VMS Deposit Style and Hydrothermal Alteration
Deposits within the BMC belong predominantly to the bimodal-siliciclastic volcanogenic massive sulphide (VMS) subtype (Galley et al., 2007; Franklin et al., 2005). Mineralization is concentrated within the Tetagouche and California Lake groups and is commonly localized near felsic volcanic-sedimentary contacts. Typical deposits comprise a chlorite-silica-pyrite stockwork developed within a sub-seafloor hydrothermal feeder system, overlain by a stratiform massive sulphide lens dominated by pyrite, sphalerite, and galena, which hosts the bulk of the Zn-Pb-Ag-Au resource. Distal hydrothermal plume activity is commonly represented by jasperous or magnetite-rich iron formations (Goodfellow, 2007).

Hydrothermal alteration forms extensive halos surrounding mineralization, progressing outward from a proximal quartz-chlorite-pyrite assemblage to sericite-carbonate-pyrite alteration. These alteration zones are important exploration vectors and are commonly associated with potassium enrichment and thorium depletion detectable in airborne radiometric datasets (Shives et al., 1997; Goodfellow, 2007).

2.3 Structural Controls and Exploration Implications
The study area is characterized by a dense network of thrust faults and subsidiary brittle structures that dissect the Ordovician volcanic belts (Fig. 1). Multiple phases of Appalachian deformation under greenschist-facies conditions have significantly modified the original geometry of the VMS systems (van Staal et al., 2003; Rogers and van Staal, 2003). Early thrusting and nappe emplacement structurally repeated favourable host sequences, while later folding and faulting further fragmented and redistributed mineralized horizons. The concentration of known mineral occurrences along major structural corridors indicates that both ore preservation and present-day exposure are strongly controlled by these tectonic processes.

The polyphase deformation history eliminates simple surface expression of mineralization, elevates cover thickness through repeated structural stacking, and makes geophysical imaging of shear zones and density contrasts an indispensable complement to geochemical sampling (Parkhill & Doiron, 2003; Thomas et al., 2000). These characteristics make integrated structural, geophysical, and geochemical approaches essential for targeting concealed VMS deposits within the Bathurst Mining Camp.

3. Methodology
Geophysical, geological, drill-hole, and till geochemistry datasets were obtained from the New Brunswick Department of Natural Resources (NBDNR) through the department's ArcGIS REST services and integrated into QGIS (v4.2.0) for visualization, quality control, and spatial data management.

The prospectivity mapping framework was structured as a multi-stage ML pipeline progressing from raw data compilation and grid-derivative computation to spatial machine learning and area-normalized validation (Fig. 2). The workflow was implemented using custom Python scripts and comprise of five key components: (1) data compilation and native-grid preprocessing; (2) compositional geochemical analysis; (3) feature extraction and engineering; (4) spatial block cross-validation and model training; and (5) full-extent mapping and interpretability.

3.1 Geophysical Datasets and Derivative Computation
Airborne geophysical grids over the BMC were used, including Total Magnetic Intensity (RTMI), Bouguer gravity, and gamma-ray spectrometric (radiometric) grids for uranium, thorium, and potassium (Figs. 3a-e). All input grids are based on the New Brunswick Stereographic Double (NAD83; EPSG:2953).

To preserve structural boundaries and prevent grid distortions caused by spatial resampling, horizontal and vertical derivatives were computed in the Fourier domain on the original survey grids prior to cell-size transformation to 100 m (Blakely, 1995):

First Vertical Derivative (FVD): Computed via multiplication by the radial wavenumber |k| in the two-dimensional Fourier domain, followed by the inverse transform, to enhance high-frequency near-surface structural and lithological contacts (Blakely, 1995):
FVDr=F-1kFk, k=kx2+ky2 (1)
where Fk=Ffr is the two-dimensional Fourier transform of the potential-field grid f, and kx, ky are the horizontal wavenumbers.

Total Horizontal Gradient (THG): Derived as the magnitude of the horizontal gradient vector, highlighting density and susceptibility contrasts at geological boundaries (Verduzco et al., 2004):
THGr=df/dx2+df/dy2 (2)
where the partial derivatives are computed via the Fourier-domain equivalents F-1ikxFk and F-1ikyFk, respectively.

Tilt Derivative (TDR): Calculated as the arctangent of the ratio of the FVD to the THG, equalizing amplitude variations between shallow and deep structural sources and providing robust edge-detection filters that delineate fault geometries and volcanic contacts (Miller & Singh, 1994):
TDRr=arctan FVD/THG (3)

Radiometric grids were preprocessed to generate radioelement ratios (K/Th, U/Th, Th/K) to map alteration zones characterized by potassic enrichment or thorium depletion indicative of VMS-related hydrothermal systems (Shives et al., 1997).

3.2 Geochemical Datasets
Till-geochemistry point data (Figs. 3f-j) were compiled from 17 separate single-element databases (Ag, As, Ba, Bi, Cd, Co, Cu, Fe, In, Mn, Mo, Ni, Pb, Sb, Sn, Tl, Zn) covering the BMC. The 17-element geochemical dataset was selected to encompass both direct indicators of VMS mineralization and broader geochemical signatures reflecting hydrothermal alteration, metal transport, and depositional processes. In addition to the principal ore metals (Zn, Pb, Cu, and Ag), the dataset includes critical-mineral pathfinder elements (In, Bi, Sn, and Sb) and elements commonly associated with sulphide-rich hydrothermal systems (Fe, Co, Ni, As, Ba, Mo, Mn, and Tl). This expanded geochemical suite allows machine-learning models to evaluate complex multivariate relationships and identify predictive signatures beyond those represented by conventional VMS pathfinders alone. Because spatial coordinates varied slightly across individual survey datasets, sample points were aligned by rounding coordinates to the nearest meter, yielding a unified point-geochemistry database of 2,753 unique locations. Inverse distance weighting (IDW) interpolation technique was used to generate geochemical surfaces (Shepard, 1968; Cardoso-Fernandes et al., 2022; McClenaghan et al., 2023):
z_hat(s0)= sum(wi*z(si))/sum(wi), wi=d(s0,si)^(-p) (4)
where z_hat(s0) is the predicted concentration at location s0, z(si) is the measured concentration at sample si, d(s0, si) is the Euclidean distance between locations, p = 2 is the power parameter, and n = 12 is the number of nearest neighbors considered.

3.3 Compositional Geochemical Analysis
To address the closed nature of compositional geochemical data, concentration values were transformed using the centered-log ratio (CLR) transformation. The CLR projects variables from the constrained simplex space into unbounded real space relative to the geometric mean of the composition (Aitchison, 1986; Egozcue et al., 2003; Filzmoser et al., 2018):
clr(x) = [ln(x1/g(x)), ln(x2/g(x)), ..., ln(xD/g(x))] (5)
where g(x) = (prod(xi))^(1/D) is the geometric mean of the D geochemical elements. Compositional PCA and compositional FA with varimax rotation were applied to the CLR-transformed IDW surfaces to extract orthogonal multi-element associations representing primary lithological units and hydrothermal alteration footprints (Filzmoser et al., 2009). PCA was used to summarize dominant geochemical variance into a reduced set of orthogonal components, whereas FA was used to identify latent multi-element associations potentially related to lithological and hydrothermal processes. The use of both methods provides complementary representations of regional geochemical variability for machine-learning analysis.

3.4 Feature Extraction and Engineering

3.4.1 Spatial Labels and Negative-Label Representativeness
The training dataset was constructed from two primary label groups. Positive labels (Y=1) comprised 45 known VMS occurrences within the BMC (van Staal et al., 2003). The generation of reliable negative labels is a fundamental challenge in mineral prospectivity mapping because the absence of a known deposit does not guarantee the absence of mineralization, particularly in incompletely explored regions (Carranza and Laborte, 2015). Although studies adapted from species distribution modelling frequently describe background training samples as pseudo-absences (Barbet-Massin et al., 2012), we adopt the term negative labels (Y = 0) to remain consistent with binary supervised classification terminology and to avoid asserting unverified geological absence. To reduce uncertainty in negative-label selection, the negative class was assembled using a hybrid class-label strategy that combined two distinct sources of evidence:

Confirmed barren drill holes (n = 125): Exploration drill intercepts compiled from New Brunswick Geological Survey records that did not intersect economic VMS mineralization. These samples provide geologically verified barren environments and anchor the negative class to locations with documented exploration results (Nykanen et al., 2008).

Feature-space dissimilar negative labels (n = 125): Candidate locations were generated on a dense 100 m grid across the active geophysical survey footprint and ranked according to their Mahalanobis distance from the centroid of the positive class in multidimensional geophysical and geochemical feature space (Mahalanobis, 1936; Carranza, 2008). Mahalanobis distance was selected in preference to unweighted metrics such as Euclidean distance because it accounts for the covariance structure of the positive class, thereby reducing the influence of correlated predictor variables and providing a more robust measure of feature-space dissimilarity. This approach operationalizes the feature-space dissimilarity framework proposed by Parsa and Cumani (2025), that negative labels which are selected to be maximally dissimilar to known deposits in predictor space rather than simply distant in geographic space improve classifier discrimination and exploration-targeting efficiency. The Mahalanobis distance from each candidate point c to the deposit centroid mu+ in standardised feature space is:
DM(c,+) = (c - mu+)^T * Sigma+^(-1) * (c - mu+) (6)
Where Sigma+^(-1) is the inverse covariance matrix of the standardised positive-class feature vectors. Candidates are ranked in descending order of DM; the 125 most dissimilar points are selected via stratified spatial sampling across four geographic quadrants to ensure geological dissimilarity is achieved without spatial clustering in any single zone of the survey footprint. A secondary minimum geographic guard distance of d = 1000 m from any known deposit is retained as a hard constraint to prevent labelling points at the immediate margins of deposit footprints, but this geographic constraint is secondary and subordinate to the feature-space dissimilarity criterion. A fixed random seed (seed = 42) was applied to ensure reproducibility (Roberts et al., 2017).

Features were extracted at the 295 training locations by sampling all geophysical derivative rasters and IDW-interpolated geochemical surfaces (Table 1). To preserve localized geochemical anomalies, raw elemental concentrations were incorporated through a nearest-neighbour spatial join. For each labelled location, the closest till-geochemistry sample within a maximum search radius of 1,000 m was identified, and the corresponding elemental concentrations were appended directly to the predictor matrix.

Table 1: The labelled dataset and predictor feature matrix

Labelled Dataset | Predictor Feature
Samples: 295 | Raw geochemistry: 17
Positive labels (VMS): 45 | Log-transformed geochemistry: 17
Negative labels: 250 | IDW geochemistry (raw) + 4 PCA + 4FA Factors: 25
Class imbalance ratio: 1:5.6 | IDW geochemistry (log): 17
 | Radiometric ratios (K/Th, U/Th & Th/K): 3
 | Geophysical features: 18
 | Composite score (MEAS): 1
 | Spatial/identity attributes: 3
 | Target: 1
 | Total: 102

3.4.2 Secondary Feature Engineering
Secondary features were engineered to capture additional mineralization criteria:

Analytic Signal (AS): Computed for both magnetics and gravity as the total amplitude of the gradient vector to isolate anomaly centres regardless of magnetization or polarization direction (Roest et al., 1992; Pham et al., 2022):
AS(r) = sqrt(THG^2 + FVD^2) (7)

Log-transformations: Applied to all raw and IDW-interpolated geochemical concentration columns to stabilize variance and normalize the right-skewed frequency distributions characteristic of trace-element geochemistry (Reimann et al., 2008):
xi' = ln(xi + 1) (8)
where xi is the raw elemental concentration and the shift of +1 prevents undefined values at zero-concentration observations.

Multi-Element Anomaly Score (MEAS): A geologically weighted composite indicator was calculated to capture anomalous concentrations of VMS pathfinder elements following the general principles of multivariate geochemical anomaly analysis outlined by Carranza, (2008). Because VMS mineralization is characterized by the co-occurrence of multiple pathfinder elements, the MEAS was used to represent their collective enrichment within a single predictor. The objective was to enhance the expression of hydrothermal geochemical signatures while reducing reliance on individual elemental anomalies. Pathfinder concentrations were transformed so that each feature has a mean of 0 and a variance of 1 and weighted (wi) according to their diagnostic association with massive sulphide mineralization.
MEAS = sum(wi * scale(xi)) (9)
where MEAS is the multi-element anomaly score, p is the number of selected pathfinder elements, xi is concentration of pathfinder element i, scale(xi) = ((xi - mu_i)/sigma_i) is the standardized value of xi and wi is the weight assigned to element i; whereas mu_i and sigma_i are the mean and standard deviation of element i respectively.

3.5 Data Quality Filtering and Class Balancing
All 17 raw geochemical elements were retained as predictor variables, as none exceeded the 75% missing-data threshold at labelled sample locations. The sparsest variables, Bi (60.3%), In (60.3%), Tl (60.3%), and Mn (58.3%), were preserved as predictors and complemented by their spatially complete IDW-interpolated surfaces, which provided values for all 295 labelled locations (Table 1). Missing values in the retained sparse features were imputed with column-wise medians, calculated solely from the training folds to avoid data leakage. This approach is robust to non-normal distributions and is widely applied in geoscientific datasets containing sparse geochemical variables (Carranza & Laborte, 2015; Reimann et al., 2008).

The final training dataset comprised 45 positive labels representing known VMS deposits and 250 negative labels, resulting in a class ratio of approximately 1:5.6 (Table 1). This class imbalance risks skewing machine-learning models in favor of the majority class, limiting their capacity to accurately detect mineralized environments (Li et al., 2020). To address this, the Synthetic Minority Over-sampling Technique (SMOTE; Chawla et al., 2002; Nidhi et al., 2026) was used to oversample the minority (positive) class during training. SMOTE generates synthetic positive instances by interpolating between neighboring minority-class samples in feature space:
x_new = xi + lambda*(x_neighbour - xi), lambda ~ U(0,1) (10)
where xi is a minority-class sample, x_neighbour is one of its nearest minority neighbors, and lambda is a random value between 0 and 1. Application of SMOTE is geologically reasonable in VMS prospectivity mapping because mineralized systems are commonly associated with continuous physical and geochemical gradients expressed through hydrothermal alteration halos, pathfinder-element dispersion patterns, and geophysical anomaly responses. Interpolation between known mineralized samples therefore generates synthetic observations that occupy plausible intermediate regions of the prospectivity feature space rather than arbitrary locations.

To prevent spatial data leakage, SMOTE was applied within the training subset of each spatial block cross-validation fold and not to the validation data. The minority class was augmented to match the size of the negative class (n = 250 per class), yielding a balanced training dataset of N = 500 samples. This procedure improved representation of mineralized environments during model training while ensuring that performance metrics were evaluated using geographically independent observations.

3.6 Spatial Block Cross-Validation
To address spatial autocorrelation and prevent overly optimistic performance estimates arising from geographically proximate observations, a 5-fold spatial block cross-validation scheme was implemented (Brenning, 2012). The study area was divided into five distinct geographic blocks (folds) containing approximately equal numbers of points/samples using quantiles of sample easting coordinates (Fig. 4). In each iteration, one fold was held out for testing while the other four were used for training, maintaining geographic separation between training and validation data (Roberts et al., 2017). This provides a more realistic evaluation of model performance in unexplored regions. Class balancing was applied separately within each training fold, following spatial partitioning of the dataset. This ensured synthetic samples were derived from training data only, while keeping validation observations spatially independent and unaffected by the balancing process.

3.7 Classifiers and Hyperparameter Tuning
Random Forest (RF; Breiman, 2001; Rodriguez-Galiano et al., 2014; Sun, et al. 2019; Mami Khalifani, 2025) and Extreme Gradient Boosting (XGBoost; Chen & Guestrin, 2016; Parsa, 2021; Ghane et al., 2026) classifiers were trained and optimized using randomized hyperparameter search within the spatial cross-validation framework. RF was implemented using the Gini impurity criterion (Breiman, 2001) to recursively partition the predictor space into homogeneous classes, whereas XGBoost was trained using a binary logistic objective function and gradient boosting optimization (Chen & Guestrin, 2016). Hyperparameter optimization was performed to identify model configurations that maximized predictive performance while reducing the risk of overfitting. Class weights were balanced in both classifiers to further mitigate residual effects of class imbalance.

3.8 Performance Metrics Evaluation
Model performance was assessed using four complementary metrics: Receiver Operating Characteristic Area Under the Curve (ROCAUC), Average Precision (AP), Balanced Accuracy (BA), and Success Rate Area Under the Curve (SRAUC) (Davis & Goadrich, 2006; Fawcett, 2006; Nykanen et al., 2015; Sun, et al. 2019; Parsa & Carranza, 2021; Nidhi et al., 2026).

3.8.1 Receiver Operating Characteristics (ROCAUC)
ROCAUC evaluates the ability of a classifier to discriminate between mineralized and non-mineralized locations across all probability thresholds by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR). It is given by:
ROCAUC = integral from 0 to 1 of TPR(FPR) d(FPR) (11)
where TPR(FPR) is the ROC curve. As a threshold-independent metric, ROCAUC measures discriminatory power across the full operating range of the classifier, making it insensitive to any particular decision boundary (Fawcett, 2006) and is widely used in mineral prospectivity mapping studies.

3.8.2 Average Precision (AP)
Average Precision evaluates the precision-recall trade-off and is particularly informative for imbalanced datasets where positive observations represent only a small fraction of the total sample population:
AP = integral from 0 to 1 of P(R) d(R) (12)
where P = Precision and R = Recall. AP penalizes models that produce excessive false positives at high recall (Davis & Goadrich, 2006). It emphasizes the ability of a model to recover positive samples while minimizing false-positive predictions.

3.8.3 Balanced Accuracy (BA)
BA provides a useful threshold-specific measure of performance for balanced training datasets generated through SMOTE. It equally weighs sensitivity and specificity of the model's ability to correctly identify both VMS deposits and barren locations and is insensitive to class imbalance. Unlike standard accuracy, it is insensitive to class imbalance and will equal 0.5 for a no-skill classifier regardless of class frequencies (Brodersen et al., 2010). Balanced Accuracy was calculated as:
BA = (1/2) * [TP/(TP+FN) + TN/(TN+FP)] (13)
BA is particularly informative here because SMOTE-balanced training sets could in principle create a model that over-predicts the positive class; a BA close to 0.7 confirms that neither class dominates the predictions at the default threshold.

3.8.4 Success Rate (SRAUC)
Evaluates targeting efficiency by plotting the cumulative fraction of known VMS deposits captured (fd) against the cumulative fraction of total study area covered (fa) when cells are ranked by prospectivity index in descending order (Carranza, 2008). It is calculated by:
SRAUC = integral from 0 to 1 of fd(fa) dfa (14)
An SRAUC value of 0.5 indicates random targeting performance, whereas a value of 1.0 represents perfect ranking of mineralized locations. SRAUC directly quantifies the economic efficiency of the model for drill-targeting decisions by measuring how much of the deposit inventory is captured within a minimal search area.

3.9 Full-Extent Mapping and Model Interpretability
Following model training and spatial cross-validation, both the Random Forest (RF) and XGBoost classifiers were projected across the full Bathurst Mining Camp prediction grid (953 x 1,253 cells; 1,194,109 total cells) to generate continuous prospectivity surfaces at 100 m spatial resolution. Prospectivity values represent the predicted probability of VMS mineralization for each grid cell, producing camp-scale maps suitable for comparison of spatial prediction patterns and delineation of prospective target areas. The resulting RF and XGBoost prospectivity maps were subsequently compared, and the classifier demonstrating the strongest performance under spatial cross-validation was recommended as the preferred model for exploration targeting.

To enhance model transparency and facilitate geological interpretation of model predictions, SHapley Additive exPlanations (SHAP; Lundberg & Lee, 2017; Ghane et al., 2026; Luo et al., 2026; Nidhi et al., 2026) were applied to both RF and XGBoost models. SHAP quantifies the contribution of individual predictors to model outputs by assigning feature-attribution values based on their marginal contribution across all possible feature combinations. Predictor importance was assessed using mean absolute SHAP values computed in probability space, enabling direct comparison of predictor influence between the two classifiers. The resulting SHAP values were used to identify the geophysical, radiometric, and geochemical variables that exert the strongest control on modelled VMS prospectivity and to support geological interpretation of the underlying mineral system.

4. Results
Results are presented for the compositional geochemical analyses, machine-learning model performance, feature-importance evaluation, and camp-scale prospectivity mapping. Comparative performance of the Random Forest and XGBoost classifiers is assessed using spatial block cross-validation metrics, followed by interpretation of the resulting prospectivity predictions.

4.1 Multivariate Geochemical Association
Principal Component Analysis (PCA) and Factor Analysis (FA) of the CLR-transformed till geochemistry dataset identified four major geochemical associations that capture distinct patterns of elemental covariance (Table 2). The dominant association, represented by PC1/FA1, is characterized by elevated Zn, Pb, Co, Ni, Sb, Cu, Ba, and Fe, together with relative depletion of In and Mo. This component reflects the principal polymetallic VMS geochemical signature within the Bathurst Mining Camp and exhibited the strongest association with known VMS mineralization among the derived multivariate variables.

The second association (PC2/FA2) is dominated by Bi and Cd enrichment and contrasts with lower In and Mo concentrations, defining a distinct geochemical population that is largely independent of the primary base-metal assemblage. PC3 expresses an Ag-enriched and As-depleted association, whereas PC4 is characterized by Sn enrichment coupled with Ag depletion. Similar patterns were identified through factor analysis, which produced broadly equivalent geochemical groupings following varimax rotation.

Factor Analysis further highlighted geochemically distinct associations within the dataset. FA3 contrasts Cu-As enrichment with Ba-Tl depletion, while FA4 is characterized by Bi enrichment and relative depletion of Ag, Mo, and Sn. Notably, FA4 subsequently emerged as one of the most influential composite geochemical predictors in the machine-learning models.

The PCA and FA results demonstrate that the till geochemistry dataset contains multiple orthogonal elemental associations representing distinct geochemical processes. These multivariate variables provide a compact representation of complex geochemical relationships and form important predictor inputs for subsequent VMS prospectivity modelling.

Table 2: The relationships and correlation strengths between the geochemical elements and the four principal components (PC) and factors (FA)

Element | PC1   | PC2   | PC3   | PC4   | FA1   | FA2   | FA3   | FA4
Ag      | 0.098 |-0.085 | 0.664 |-0.528 | 0.339 |-0.043 |-0.101 |-0.362
As      | 0.251 | 0.095 |-0.360 |-0.142 | 0.637 | 0.449 | 0.311 | 0.237
Ba      | 0.271 |-0.023 | 0.236 | 0.190 | 0.779 | 0.345 |-0.468 | 0.073
Bi      |-0.002 | 0.580 | 0.032 | 0.186 | 0.321 |-0.603 | 0.257 | 0.513
Cd      |-0.093 | 0.523 |-0.038 |-0.087 | 0.025 |-0.625 | 0.229 | 0.264
Co      | 0.298 |-0.134 |-0.124 |-0.063 | 0.712 | 0.664 |-0.009 |-0.079
Cu      | 0.274 |-0.060 |-0.340 |-0.184 | 0.642 | 0.641 | 0.364 |-0.023
Fe      | 0.269 |-0.219 |-0.055 |-0.023 | 0.617 | 0.615 |-0.113 |-0.238
In      |-0.219 |-0.339 |-0.357 |-0.158 |-0.947 | 0.311 |-0.014 | 0.028
Mn      | 0.292 | 0.167 | 0.022 | 0.045 | 0.923 | 0.224 | 0.111 | 0.043
Mo      |-0.204 |-0.329 | 0.190 | 0.228 |-0.588 |-0.226 |-0.235 |-0.449
Ni      | 0.285 |-0.017 |-0.086 |-0.107 | 0.776 | 0.452 | 0.219 |-0.149
Pb      | 0.302 |-0.005 | 0.036 | 0.078 | 0.843 | 0.419 |-0.105 | 0.040
Sb      | 0.279 |-0.108 | 0.076 |-0.141 | 0.728 | 0.459 |-0.057 |-0.159
Sn      | 0.139 |-0.183 | 0.021 | 0.639 | 0.362 | 0.185 |-0.081 |-0.257
Tl      | 0.266 | 0.004 | 0.237 | 0.244 | 0.777 | 0.296 |-0.438 | 0.134
Zn      | 0.307 | 0.088 |-0.054 |-0.014 | 0.873 | 0.419 | 0.029 | 0.151

4.2 Model Performance Under Spatial Cross-Validation
Spatial block cross-validation (Fig. 4) results demonstrate strong predictive performance for both the Random Forest (RF) and XGBoost models (Table 3). RF achieved the highest overall discrimination and targeting performance, with a mean ROCAUC of 0.9318 +/- 0.0368, Average Precision of 0.7245 +/- 0.1476, and Success Rate of 0.9680, compared with 0.9098 +/- 0.0369, 0.6226 +/- 0.1481, and 0.9494, respectively, for XGBoost. XGBoost produced a marginally higher Balanced Accuracy (0.8456 +/- 0.0654) than RF (0.8261 +/- 0.0701).

Performance variability across the five spatial folds was low for both classifiers, with nearly identical ROC-AUC standard deviations (RF: +/-0.0368; XGBoost: +/-0.0369), indicating consistent predictive behaviour across geographically independent validation blocks. Overall, RF provided the strongest combination of discrimination and exploration-targeting performance (Fig. 5).

Table 3. Mean spatial block cross-validation performance metrics.

Metric | Random Forest | XGBoost
Receiver Operating Characteristics, ROCAUC (mean +/- standard deviation) | 0.9318 +/- 0.0368 | 0.9098 +/- 0.0369
Average Precision (mean +/- standard deviation) | 0.7245 +/- 0.1476 | 0.6226 +/- 0.1481
Balanced Accuracy (mean +/- standard deviation) | 0.8261 +/- 0.0701 | 0.8456 +/- 0.0654
Success Rate, SRAUC | 0.968 | 0.949

4.3 VMS Prospectivity Patterns Across the Bathurst Mining Camp
The prospectivity maps generated by the Random Forest (RF) and XGBoost classifiers reveal broadly similar spatial patterns of predicted VMS prospectivity across the Bathurst Mining Camp (Fig. 6). In both models, elevated Prospectivity Index (PI) values form spatially coherent, elongate target corridors exhibiting a predominantly north-northeast to south-southwest (NNE-SSW) orientation, broadly parallel to the regional structural and stratigraphic framework of the Tetagouche Group. The highest-prospectivity zones form continuous linear belts that coincide with known VMS clusters and favourable volcanic horizons. The XGBoost model predicts a larger spatial footprint of high-prospectivity areas, whereas the RF model produces a more concentrated set of anomalies.

The RF prospectivity map exhibits a highly localized distribution of predicted mineralization potential, with PI values ranging from 0 to 1 and a median value of 0.049. High-priority targets (PI > 0.7) occupy 23,585 cells (2.0% of the study area), whereas moderate-to-high prospectivity zones (PI > 0.5) encompass 85,303 cells (7.1%). Very high-priority areas (PI > 0.9) are restricted to 2,678 cells (0.2%). In contrast, the XGBoost prospectivity map displays a lower median PI value (0.0025) and a more polarized probability distribution, delineating 59,588 cells (5.0%) exceeding PI > 0.7 and 92,200 cells (7.7%) exceeding PI > 0.5.

Despite differences in the extent and distribution of high-prospectivity areas, both classifiers identify several common prospective corridors associated with favourable volcanic stratigraphy and regional structural trends. Several anomalies with PI values exceeding 0.7 occur beyond the footprint of currently documented VMS deposits, highlighting additional prospective areas within covered portions of the camp. Since the RF map delineates a more spatially focused set of anomalies, it was therefore selected as the preferred prospectivity model based on its superior performance under spatial cross-validation (Table 3).

4.4 Key Predictors of VMS Prospectivity
Mean absolute SHAP values, computed in probability space, were used to quantify predictor importance for the Random Forest (RF) and XGBoost models (Table 4). Both classifiers showed strong agreement regarding the primary controls on VMS prospectivity. The radiometric Th/K alteration ratio emerged as the highest-ranked predictor in both RF (0.0501) and XGBoost (0.0857), highlighting the importance of alteration-related radiometric signatures. Molybdenum-related variables were consistently among the most influential predictors, with IDW-interpolated molybdenum ranking third in RF and second in XGBoost. Zinc-related variables also featured prominently in both models, further emphasizing their importance as VMS pathfinder indicators.

Although the highest-ranked predictors were broadly consistent, the two models differed in their relative emphasis on specific data domains. RF assigned greater importance to radiometric variables and interpolated geochemical surfaces, including Zn, Bi, and Pb anomaly layers, whereas XGBoost relied on a broader combination of geochemical, radiometric, gravity, and magnetic predictors. Notably, upward-continued Bouguer gravity, magnetic analytic signal, and the radiometric U/Th ratio ranked among the most influential variables in XGBoost but were comparatively less important in RF.

Table 4: Top 10 predictor features ranked by mean absolute SHAP values for the Random Forest and XGBoost models. SHAP values were calculated in probability space and reflect each predictor's average contribution to modelled VMS prospectivity.

Rank | Random Forest Feature | Mean |SHAP| | XGBoost Feature | Mean |SHAP|
1 | Radiometric Th/K alteration ratio | 0.0501 | Radiometric Th/K alteration ratio | 0.0857
2 | Radiometric thorium (Th) | 0.0397 | IDW-interpolated molybdenum (Mo) | 0.0469
3 | IDW-interpolated molybdenum (Mo) | 0.0339 | Raw zinc (Zn) | 0.0372
4 | Radiometric potassium (K) | 0.0184 | Raw molybdenum (Mo) | 0.0274
5 | IDW-interpolated zinc (Zn) | 0.0184 | IDW-interpolated tin (Sn) | 0.0262
6 | IDW-interpolated bismuth (Bi) | 0.0147 | Upward-continued Bouguer gravity | 0.0244
7 | IDW-interpolated lead (Pb) | 0.0138 | Radiometric thorium (Th) | 0.0228
8 | Raw zinc (Zn) | 0.0132 | Raw nickel (Ni) | 0.0224
9 | Gravity horizontal gradient magnitude | 0.0128 | Magnetic analytic signal | 0.0219
10 | Raw molybdenum (Mo) | 0.0126 | Radiometric U/Th ratio | 0.0205

These feature rankings identify three dominant predictor groups controlling modelled VMS prospectivity: (1) radiometric indicators of hydrothermal alteration, particularly the Th/K ratio; (2) geochemical pathfinder variables, notably Mo, Zn, Pb, Bi, Sn, and Ni; and (3) geophysical derivatives that capture lithological contrasts and structural architecture. The strong agreement between RF and XGBoost indicates that these predictor domains represent robust controls on VMS prospectivity within the Bathurst Mining Camp.

5. Discussion

5.1 Controls on VMS Prospectivity in the Bathurst Mining Camp
Feature importance analysis identified three dominant predictor domains: radiometric indicators of hydrothermal alteration, geochemical pathfinder signatures, and geophysical expressions of structural architecture. Among these, alteration-related radiometric variables emerged as the strongest controls on modelled VMS prospectivity.

The dominance of the Th/K alteration ratio is consistent with established models of VMS hydrothermal alteration, in which sericitic alteration commonly results in potassium enrichment relative to thorium within feeder zones and alteration halos (Franklin et al., 2005; Galley et al., 2007). Consequently, low Th/K ratios are commonly associated with hydrothermal alteration and can be used to trace potential fluid pathways (Shives et al., 1997). The consistently high ranking of both Th/K and thorium in the RF and XGBoost models suggests that alteration-related modification of the host rocks exerts a stronger control on prospectivity than individual geochemical pathfinders and was successfully captured by the machine-learning framework.

Geochemical predictors provide complementary evidence for mineralization. Molybdenum was among the most influential variables in both classifiers, with both raw and IDW-interpolated Mo consistently ranked within the top predictors. This association is consistent with enrichment of Mo in high-temperature hydrothermal fluids and feeder-stockwork environments commonly associated with VMS systems (Franklin et al., 2005). Zinc-related variables also ranked highly, reflecting the central role of Zn within the polymetallic signature of BMC deposits. Additional pathfinder elements, including Pb, Bi, Sn, and Ni, suggest that the models capture a broad hydrothermal metal assemblage rather than isolated elemental anomalies.

The multivariate geochemical components derived from PCA and FA further reinforce this interpretation. PC1/FA1 represents the principal polymetallic VMS association, characterized by Zn-Pb-Cu-Ba enrichment together with associated Co and Ni, and exhibits the strongest relationship with known mineralization. This assemblage is consistent with the characteristic geochemical signatures reported for VMS deposits and the Bathurst Mining Camp (Franklin et al., 2005; Galley et al., 2007; Goodfellow & McCutcheon, 2003). Secondary components, including Bi-Cd enrichment (PC2/FA2), Ag-As variability (PC3), and Sn enrichment (PC4), define distinct geochemical populations that may reflect variations in hydrothermal processes, metal distribution, or mineralizing conditions across the camp. However, their geological significance appears to be secondary to the dominant polymetallic VMS signature represented by PC1/FA1.

Geophysical derivatives also contributed significantly to prospectivity prediction. Gravity horizontal gradients, upward-continued gravity responses, magnetic analytic signal amplitudes, and magnetic first vertical derivatives highlight the importance of lithological contacts and syn-volcanic fault systems that controlled fluid migration and sulfide deposition (Van Staal et al., 2003; Thomas et al., 2000). These results indicate that the most effective predictors represent an integrated alteration-mineralization-structure framework, consistent with established geological models for BMC VMS formation.

5.2 Implications for Camp-Scale Exploration Targeting
Both machine-learning classifiers demonstrated strong predictive capability under spatially independent validation, confirming the effectiveness of integrating geophysical derivatives, multi-element till geochemistry, and geologically constrained training labels for regional-scale VMS targeting. Random Forest (RF) consistently outperformed XGBoost across ROCAUC, Average Precision, and Success Rate AUC metrics, indicating superior discrimination between mineralized and non-mineralized locations and greater efficiency in prioritizing exploration targets. The low fold-to-fold variability observed for both classifiers further indicates stable model generalization across geographically independent portions of the Bathurst Mining Camp (BMC), suggesting that the predictive relationships identified by the models are not restricted to individual deposit clusters.

The RF and XGBoost prospectivity maps provide complementary spatial syntheses of the geological, geochemical, and geophysical information represented by the predictor dataset. The NNE-SSW alignment of high-prospectivity corridors identified by both RF and XGBoost reflects the strong influence of the regional volcanic and structural architecture of the Tetagouche Group on VMS distribution. The close correspondence between these corridors and known deposit clusters supports the geological validity of the prospectivity models and suggests that additional targets occurring along the same trend represent favourable exploration opportunities. Conversely, low-prospectivity regions correspond largely to geological units that are not considered favourable hosts for bimodal-siliciclastic VMS mineralization, indicating that the prospectivity patterns are consistent with the established geological framework of the BMC (Goodfellow, 2007; van Staal et al., 2003).

The RF model achieved particularly strong targeting performance, capturing 91.1% of known VMS deposits within the top 10% of the ranked study area, 97.8% within the top 20%, and 100% within the top 30%. These results demonstrate that the integrated workflow concentrates known mineralization into a relatively small proportion of the camp, thereby substantially reducing the search space for exploration. Reducing the search area while retaining a large proportion of known deposits is a principal objective of mineral prospectivity mapping and an important measure of map performance (Agterberg & Bonham-Carter, 2005; Carranza, 2008).

Several high-prospectivity anomalies identified by both classifiers occur outside the current inventory of known deposits while remaining spatially associated with favourable structural corridors, alteration signatures, and geochemical anomalies. Although the overall prospectivity patterns are broadly similar, the RF model delineates a more spatially focused set of target areas than XGBoost, consistent with its superior targeting efficiency under spatial cross-validation. These new anomalies represent compelling targets for follow-up exploration in the region or other covered terrain where conventional geological mapping may be less effective.

The strong predictive performance achieved by both classifiers also highlights the importance of representative negative training labels in mineral prospectivity mapping. The hybrid class label strategy adopted in this study, which combines confirmed barren drill intercepts with feature-space dissimilar background samples, provided a more geologically meaningful representation of negative conditions than conventional random background sampling. This interpretation is consistent with the recent work by Parsa and Cumani (2025), who demonstrated that the representativeness of negative class labels can significantly influence classification performance and the spatial selectivity of prospectivity models. The low variability observed across spatial cross-validation folds and the high targeting efficiency of the RF model suggest that geologically constrained negative-label selection contributed to robust model performance and improved camp-scale exploration targeting.

5.3 Limitations and Future Directions
Despite the strong model performance, several limitations are acknowledged. The modelling framework was trained using a relatively small set of known VMS deposits, reflecting the finite inventory of documented occurrences within the BMC. Although the hybrid class-label strategy improved class separation and validation stability, uncertainty inevitably remains in poorly explored areas where the true distribution of mineralization is unknown.

The prospectivity framework is also constrained by the datasets available at camp scale. Future work could also explore the integration of multiple classifier outputs through ensemble prospectivity modelling and uncertainty quantification to better characterize areas of agreement and disagreement between machine-learning predictions.

Notwithstanding these limitations, the results demonstrate that integrating radiometric alteration signatures, structural geophysical derivatives, and multi-element till geochemistry within a machine-learning framework provides an effective approach for camp-scale VMS exploration targeting in covered terranes.

6. Conclusions
This study demonstrates the effectiveness of integrating geophysical derivatives, multi-element till geochemistry, and machine learning for camp-scale volcanogenic massive sulphide (VMS) prospectivity mapping in the Bathurst Mining Camp (BMC), New Brunswick. Spatially independent validation showed that both Random Forest (RF) and XGBoost successfully captured the principal geological controls on VMS mineralization, confirming the value of combining alteration, geochemical, and structural datasets within a unified prospectivity framework.

Among the evaluated classifiers, RF provided the strongest overall exploration-targeting performance, achieving a ROCAUC of 0.9318 +/- 0.0368 and a success-rate AUC of 0.968. Furthermore, 91.1% of known VMS deposits were captured within the highest-ranked 10% of the study area, indicating that the model can substantially reduce the exploration search space while maintaining high deposit recovery.

Feature importance analysis revealed strong agreement between RF and XGBoost regarding the principal controls on prospectivity. Radiometric indicators of hydrothermal alteration, particularly the Th/K ratio, emerged as the most influential predictors, followed by molybdenum- and zinc-related geochemical variables and gravity- and magnetic-derived structural attributes. Together, these findings highlight the fundamental roles of hydrothermal alteration, metal dispersion, and structural architecture in controlling the distribution of VMS mineralization within the BMC.

Prospectivity maps generated by both RF and XGBoost delineate several high-priority targets beyond the current inventory of known deposits while remaining consistent with established geological, geochemical, and structural controls. Despite broadly similar spatial patterns, RF produced a more focused distribution of high-prospectivity zones and superior exploration-targeting performance, supporting its recommendation as the preferred model for camp-scale VMS exploration in the BMC.

Importantly, we have demonstrated the value of a geologically constrained hybrid class-label framework that combines confirmed barren drill intercepts with feature-space dissimilar background samples to represent non-mineralized conditions. The overall results show that integrating geophysical derivatives, multi-element till geochemistry, and geologically informed class labels within a spatially validated machine-learning workflow provides an effective and transferable approach for camp-scale VMS exploration targeting in mature and partially covered mining districts.

Figure Captions
Fig. 1 The geology of the BMC with major lithostratigraphic units, structural trends and locations of known VMS deposits (e.g. Brunswick No. 12 (B12) and Brunswick No. 6 (B6)) deposits. Inset: Approximate location of the BMC, northern New Brunswick, Canada. Modified from van Staal et al. (2003).

Fig. 2 Schematic workflow chart of the machine learning prospectivity mapping pipeline. Boxes denote processing stages; arrows denote data flow. CLR centered log-ratio; CoDA compositional data analysis; FA factor analysis; IDW inverse distance weighting; MEAS multi-element anomaly score; PCA principal component analysis; SMOTE synthetic minority over-sampling technique; RF random forest; XGBoost extreme gradient boosting; and SHAP SHapley Additive exPlanations.

Fig. 3 Geophysical and geochemical input datasets for the BMC obtained from NBDNR through ArcGIS REST services; (a-e) Airborne residual total magnetic intensity (RTMI), Bouguer gravity, uranium (eU ppm), thorium (eTh ppm) and potassium (%K); (f-j) Selected till-geochemistry point data for lead (Pb ppm), zinc (Zn ppm), copper (Cu ppm), silver (Ag) and antimony (Sb).

Fig. 4. Spatial cross-validation fold assignment used for model evaluation. Labelled locations were partitioned into five geographically independent folds based on easting-coordinate quantiles, yielding approximately equal sample counts per fold. In each cross-validation iteration, one fold was reserved for validation while the remaining four folds were used for model training. The residual magnetic intensity layer shown in the background provides geological context and illustrates the spatial distribution of validation folds across the Bathurst Mining Camp.

Fig. 5. Success-rate curves for the Random Forest (RF) and XGBoost classifiers under spatial cross-validation. Curves show the cumulative percentage of known VMS deposits captured as a function of the cumulative percentage of the study area ranked by prospectivity. The RF model demonstrates superior targeting efficiency, capturing 91.1% of known deposits within the highest-ranked 10% of the study area and achieving a larger Success Rate AUC than XGBoost.

Fig. 6. Camp-scale VMS prospectivity maps generated using (a) Random Forest and (b) XGBoost models. Prospectivity Index (PI) values represent the predicted probability of VMS mineralization. Dotted circles indicate known VMS deposits, and outlined polygons represent high-priority target areas (PI > 0.7). Both models delineate NNE-SSW trending prospective corridors. The Random Forest model produces a more spatially focused distribution of high-prospectivity zones.

References
Agterberg, F.P., Bonham-Carter, G.F. (2005). Measuring the Performance of Mineral-Potential Maps. Natural Resources Research 14, 1-17. https://doi.org/10.1007/s11053-005-4674-0

Aitchison, J. (1986). The statistical analysis of compositional data. Chapman and Hall.

Barbet-Massin, M., Jiguet, F., Albert, C. H., & Thuiller, W. (2012). Selecting pseudo-absences for species distribution models: how, where and how many? Methods in Ecology and Evolution, 3(2), 327-338. https://doi.org/10.1111/j.2041-210X.2011.00172.x

Blakely, R. J. (1995). Potential theory in gravity and magnetic applications. Cambridge University Press.

Bonham-Carter, G. F. (1994). Geographic information systems for geoscientists: Modelling with GIS. Pergamon Press.

Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

Brenning, A. (2012). Spatial cross-validation and bootstrap for the assessment of prediction rules in remote sensing: The R package sperrorest. 2012 IEEE International Geoscience and Remote Sensing Symposium, 5372-5375. https://doi.org/10.1109/IGARSS.2012.6352393

Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. In Proceedings of the 20th International Conference on Pattern Recognition (ICPR) (pp. 3121-3124). IEEE. https://doi.org/10.1109/ICPR.2010.764

Cardoso-Fernandes, J., Lima, J., Lima, A., Roda-Robles, E., Kohler, M., Schaefer, S., Barth, A., Knobloch, A., Goncalves, M. A., Goncalves, F., & Teodoro, A. C. (2022). Stream sediment analysis for Lithium (Li) exploration in the Douro region (Portugal): A comparative study of the spatial interpolation and catchment basin approaches. Journal of Geochemical Exploration, 236, 106978. https://doi.org/10.1016/j.gexplo.2022.106978

Carranza, E.J.M. (2008). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS. Handbook of Exploration and Environmental Geochemistry, Vol. 11. Elsevier, Amsterdam.

Carranza, E. J. M., & Laborte, A. G. (2015). Random forest predictive modeling of mineral prospectivity with small number of prospects and data with missing values in Abra (Philippines). Computers & Geosciences, 74, 60-70. https://doi.org/10.1016/j.cageo.2014.10.004

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794. https://doi.org/10.1145/2939672.2939785

Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In Proceedings of the 23rd International Conference on Machine Learning (ICML '06) (pp. 233-240). ACM. https://doi.org/10.1145/1143844.1143874

Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barcelo-Vidal, C. (2003). Isometric logratio transformations for compositional data analysis. Mathematical Geology, 35(3), 279-300. https://doi.org/10.1023/A:1023818214614

Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874. https://doi.org/10.1016/j.patrec.2005.10.010

Filzmoser, P., Hron, K., & Reimann, C. (2009). Principal component analysis for compositional data with outliers. Environmetrics, 20(6), 621-632. https://doi.org/10.1002/env.966

Filzmoser, P., Hron, K., & Templ, M. (2018). Applied Compositional Data Analysis: With R examples. Springer. https://doi.org/10.1007/978-3-319-96422-5

Franklin, J. M., Gibson, H. L., Jonasson, I. R., & Galley, A. G. (2005). Volcanogenic massive sulphide deposits. Economic Geology, 100th Anniversary Volume, 523-560. https://doi.org/10.5382/AV100.17

Galley, A. G., Hannington, M. D., & Jonasson, I. R. (2007). Volcanogenic massive sulphide deposits. In W. D. Goodfellow (Ed.), Mineral deposits of Canada: A synthesis of major deposit-types, district metallogeny, the evolution of geological provinces, and the exploration methods (Special Publication No. 5, pp. 141-161). Geological Association of Canada, Mineral Deposits Division.

Ghane, B., Lentz, D.R., Thorne, K.G., Ugalde, H. A., (2026). Calibration of Airborne Geophysical Data with In Situ Petrophysical Measurements for Mineral Prospectivity Mapping Using XGBoost Random Forest. Nat Resour Res 35, 245-277 (2026). https://doi.org/10.1007/s11053-025-10579-7

Goodfellow, W. D. (2007). Metallogeny of the Bathurst Mining Camp, northern New Brunswick. In W. D. Goodfellow (Ed.), Mineral deposits of Canada: A synthesis of major deposit-types, district metallogeny, the evolution of geological provinces, and the exploration methods (Special Publication No. 5, pp. 443-469). Geological Association of Canada, Mineral Deposits Division.

Goodfellow, W. D., & McCutcheon, S. R. (2003). Geologic and genetic attributes of volcanic-associated massive sulphide deposits of the Bathurst Mining Camp, northern New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine (Economic Geology Monograph No. 11, pp. 19-60). Society of Economic Geologists. https://doi.org/10.5382/Mono.11.13

Li, T., Xia, Q., Zhao, M., Gui, Z., & Leng, S. (2020). Prospectivity mapping for tungsten polymetallic mineral resources, Nanling Metallogenic Belt, South China: Use of Random Forest algorithm from a perspective of data imbalance. Natural Resources Research, 29(1), 203-227. https://doi.org/10.1007/s11053-019-09564-8

Lundberg, S. M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

Luo, Z., Farahbakhsh, E., Hore, S., and Muller, R. D. (2026). DEEP-SEAM: an explainable semi-supervised deep learning framework for mineral prospectivity mapping, Geosci. Model Dev., 19, 2593-2625, https://doi.org/10.5194/gmd-19-2593-2026.

Maepa, F., Smith, R. S., & Tessema, A. (2021). Support vector machine and artificial neural network modelling of orogenic gold prospectivity mapping in the Swayze greenstone belt, Ontario, Canada. Ore Geology Reviews, 139, 104408. https://doi.org/10.1016/j.oregeorev.2020.103968

Mahalanobis, P.C. (1936). On the generalized distance in statistics. Proceedings of the National Institute of Sciences of India, 2, 49-55.

Mami Khalifani, F., Lentz, D.R. & Walker, J.A. (2025) Machine learning-based mineral prospectivity mapping of epithermal gold mineralization in Northern New brunswick: a comparative study of random forest, support vector machine, and XGboost classifiers. Earth Sci Inform 18, 553. https://doi.org/10.1007/s12145-025-02041-2

McClenaghan, M. B., Paulen, R. C., Smith, I. R., Rice, J. M., Plouffe, A., McMartin, I., Campbell, J. E., Lehtonen, M., Parsasadr, M., & Beckett-Brown, C. E. (2023). Review of till geochemistry and indicator mineral methods for mineral exploration in glaciated terrain. Geochemistry: Exploration, Environment, Analysis, 23(4), geochem2023-013. https://doi.org/10.1144/geochem2023-013

McCutcheon, S. R., & Walker, J. A. (2020). Great mining camps of Canada 8. The Bathurst Mining Camp, New Brunswick, Part 2: Mining history and contributions to society. Geoscience Canada, 47(3), 143-166. https://doi.org/10.12789/geocanj.2020.47.163.

Miller, H. G., & Singh, V. (1994). Potential field tilt-A new concept for location of potential field sources. Journal of Applied Geophysics, 32(2-3), 213-217. https://doi.org/10.1016/0926-9851(94)90022-1

Natural Resources Canada. (2024). Canada's critical minerals list 2024. Government of Canada. https://www.canada.ca/en/natural-resources-canada/news/2024/06/government-of-canada-releases-updated-critical-minerals-list.html. Accessed 2026.

Nidhi, D.K., Mohapatra, S.K., Nevalainen, P., Heikkonen, J., & Kanth, R.,(2026). Mineral prospectivity mapping under extreme imbalance using contrastive embeddings, balanced learning and integrated uncertainty analysis. Discov Computing 29, 288. https://doi.org/10.1007/s10791-026-10192-z

Nykanen, V., Groves, D. I., Ojala, V. J., Eilu, P., & Gardoll, S. J. (2008). Reconnaissance-scale conceptual fuzzy-logic prospectivity modelling for iron oxide copper-gold deposits in the northern Fennoscandian Shield, Finland. Australian Journal of Earth Sciences, 55(1), 25-38. https://doi.org/10.1080/08120090701581372

Nykanen, V., Lahti, I., Niiranen, T., & Korhonen, K. (2015). Receiver operating characteristics (ROC) as validation tool for prospectivity models-A magmatic Ni-Cu case study from the Central Lapland Greenstone Belt, Northern Finland. Ore Geology Reviews, 71, 853-860. https://doi.org/10.1016/j.oregeorev.2014.09.007

Parkhill, M. A., & Doiron, A. (2003). Quaternary geology and till geochemistry of the Bathurst Mining Camp, New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine (Economic Geology Monograph No. 11, pp. 101-122). Society of Economic Geologists https://doi.org/10.5382/Mono.11.28

Parsa, M. (2021). A data augmentation approach to XGboostbased mineral potential mapping: An example of carbonate hosted Zn-Pb mineral systems of Western Iran. Journal of Geochemical Exploration, 228, 106811. https://doi.org/10.1016/j.gexplo.2021.106811

Parsa, M., & Carranza, E. J. M. (2021). Modulating the impacts of stochastic uncertainties linked to deposit locations in data-driven predictive mapping of mineral prospectivity. Natural Resources Research, 30(5), 3081-3097. https://doi.org/10.1007/s11053-021-09891-9

Parsa, M., Lentz, D. R., & Walker, J. A. (2023). Predictive modeling of prospectivity for VHMS mineral deposits, northeastern Bathurst Mining Camp, NB, Canada, using an ensemble regularization technique. Natural Resources Research, 32, 19-36. https://doi.org/10.1007/s11053-022-10133-9

Parsa, M., Cumani, R. (2025). Class Label Representativeness in Machine Learning-Based Mineral Prospectivity Mapping. Natural Resources Research 34, 1901-1925. https://doi.org/10.1007/s11053-025-10468-z

Pham, L. T., Eldosouky, A. M., Oksum, E., & Saada, S. A. (2022). A new high resolution filter for source edge detection of potential field data. Geocarto International, 37(11), 3051-3068. https://doi.org/10.1080/10106049.2020.1849414

Reimann, C., Filzmoser, P., Garrett, R. G., & Dutter, R. (2008). Statistical data analysis explained: Applied environmental statistics with R. Wiley. https://doi.org/10.1002/9780470987605

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schroder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, or phylogenetic structure. Ecography, 40(8), 913-929. https://doi.org/10.1111/ecog.02881

Rodriguez-Galiano, V. F., Chica-Olmo, M., & Chica-Rivas, M. (2014). Predictive modelling of gold potential with the integration of multisource information based on random forest: a case study on the Rodalquilar area, Southern Spain. International Journal of Geographical Information Science, 28(7), 1336-1354. https://doi.org/10.1080/13658816.2014.885527

Roest, W. R., Verhoef, J., & Pilkington, M. (1992). Magnetic interpretation using the 3-D analytic signal. Geophysics, 57(1), 116-125. https://doi.org/10.1190/1.1443174

Rogers, N., & van Staal, C. R., (2003). Volcanology and Tectonic Setting of the Northern Bathurst Mining Camp: Part II. Mafic Volcanic Constraints on Back-Arc Opening. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine (Economic Geology Monograph No. 11, pp. 61-100). Society of Economic Geologists. https://doi.org/10.5382/Mono.11.10

Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. In Proceedings of the 1968 23rd ACM National Conference (pp. 517-524). ACM. https://doi.org/10.1145/800186.810616

Shives, R.B.K., Charbonneau, B.W., and Ford, K.L., 1997, The detection of potassic alteration by gamma-ray spectrometry-recognition of alteration related to mineralization, in A.G. Gubins, ed., Proceedings of Exploration 97, Fourth Decennial International Conference on Mineral Exploration: Geophysics and Geochemistry at the Millennium, p. 741-752.

Sun, T., Chen, F., Zhong, L., Liu, W., Wang, Y. (2019). GIS-based mineral prospectivity mapping using machine learning methods: A case study from Tongling ore district, eastern China. Ore Geology Reviews 109 (2019) 26-49. https://doi.org/10.1016/j.oregeorev.2019.04.003

Thomas, M. D., Walker, J. A., Keating, P. B., Shives, R. B. K., Kiss, F. G. & Goodfellow, W. D. (2000). Geophysical atlas of massive sulphide signatures Bathurst mining camp, New Brunswick. Geological Survey of Canada, Open File, 3887, 105. https://doi.org/10.4095/211549

Ugalde, H., Morris, W. A., & van Staal, C. R. (2019). The Bathurst Mining Camp, New Brunswick: data integration, geophysical modelling, and implications for exploration. Canadian Journal of Earth Sciences, 56(5), 433-451. https://doi.org/10.1139/cjes-2018-0048

van Staal, C. R., Wilson, R. A., Rogers, N., Fyffe, L. R., Langton, J. P., McCutcheon, S. R., McNicoll, V., & Ravenhurst, C. E. (2003). Geology and Tectonic History of the Bathurst Supergroup, Bathurst Mining Camp, and Its Relationships to Coeval Rocks in Southwestern New Brunswick and Adjacent Maine-A Synthesis. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine (Economic Geology Monograph No. 11, pp. 37-60). Society of Economic Geologists. https://doi.org/10.5382/Mono.11.03

Verduzco, B., Fairhead, J. D., Green, C. M., & MacKenzie, C. (2004). New insights into magnetic derivatives for structural mapping. The Leading Edge, 23(2), 116-119. https://doi.org/10.1190/1.1651454

Zuo, R., Kreuzer, O.P., Wang, J., Xiong, Y., Zhang, Z. and Wang, Z., 2021. Uncertainties in GIS-based mineral prospectivity mapping: Key types, potential impacts and possible solutions. Natural Resources Research, 30(5), pp.3059-3079. https://link.springer.com/article/10.1007/s11053-021-09871-z

Zuo, R., Xiong, Y., Wang, Z., & Carranza, E. J. M. (2019). Deep learning and its application in geochemical mapping. Earth-Science Reviews, 192, 1-14. https://doi.org/10.1016/j.earscirev.2019.02.023
