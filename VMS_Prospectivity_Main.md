**Camp-Scale Machine Learning Prospectivity Mapping of VMS Deposits in the Bathurst Mining Camp, New Brunswick: Integrating Geophysical Derivatives, Multi-Element Till Geochemistry, and Geologically Constrained Class Labels**

**Dele Falebita<sup>1</sup> · Mohammad Parsa<sup>2</sup> · David Lentz<sup>3</sup>**

<sup>1</sup> Tech Connect Southeast, Venn Innovation, Moncton, New Brunswick, Canada

<sup>2</sup> Natural Resources Canada, Geological Survey of Canada, Ottawa, Ontario, Canada

<sup>3</sup> Department of Earth Sciences, University of New Brunswick, Fredericton, New Brunswick, Canada

**Corresponding author:** Dele Falebita, [dele@technb.ca](mailto:dele@technb.ca); [dele.1.falebita@gmail.com](mailto:dele.1.falebita@gmail.com); [ORCID](https://orcid.org/0000-0002-8154-5832)

**Acknowledgement**

The authors appreciate the New Brunswick Department of Natural Resources for making the data available through their ArcGIS REST server.

**Author Contributions**

Dele Falebita performed the data analysis, interpreted the results, and prepared the original manuscript draft. Mohammad Parsa reviewed and edited the manuscript and provided technical guidance throughout the study. David Lentz provided supervision and oversight of the research.

**Statements and Declarations**

**Conflict of Interest:** The authors declare that they have no known financial or personal relationships that could have influenced the work reported in this paper.

**Camp-Scale Machine Learning Prospectivity Mapping of VMS Deposits in the Bathurst Mining Camp, New Brunswick: Integrating Geophysical Derivatives, Multi-Element Till Geochemistry, and Geologically Constrained Class Labels**

**Abstract**

We present a machine-learning framework for camp-scale volcanogenic massive sulphide (VMS) prospectivity mapping in the Bathurst Mining Camp (BMC), New Brunswick, Canada. Aeromagnetic, gravity, and radiometric datasets were integrated with a compiled 17-element till geochemistry dataset comprising 2,753 sample locations. Geophysical derivatives were generated to enhance structural features, while geochemical surfaces were interpolated using inverse distance weighting and transformed using centered log-ratio (CLR) methods. Principal Component Analysis (PCA), Factor Analysis (FA), and a geologically weighted Multi-Element Anomaly Score (MEAS) were used to extract geochemically meaningful predictors associated with VMS mineralization. Random Forest (RF) and Extreme Gradient Boosting (XGBoost) classifiers were trained using 295 spatial labels consisting of 45 known VMS deposits and 250 geologically constrained negative labels derived from barren drill intercepts and feature-space dissimilar samples. Model evaluation employed Synthetic Minority Over-sampling Technique (SMOTE) class balancing and 5-fold spatial cross-validation. RF outperformed XGBoost across the principal discrimination and targeting metrics, achieving Receiver Operating Characteristics-Area Under Curve (ROC<sub>AUC</sub>) values of 0.9318 ± 0.0368 and 0.9098 ± 0.0369, and Success Rate AUC values of 0.9680 and 0.9494 respectively. The RF model captured 91.1% of known VMS deposits within the highest-ranked 10% of the study area. Prospectivity maps produced by both classifiers delineated spatially coherent NNE-SSW-trending corridors that coincide with known VMS clusters and favourable Tetagouche Group volcanic horizons, while also identifying previously unrecognized target areas. The radiometric Thorium/Potassium (Th/K) alteration ratio emerged as the most influential predictor in both models, followed by molybdenum-, zinc-, and lead-related geochemical variables. The results demonstrate the value of integrating hydrothermal alteration signatures, structural geophysics, and multi-element till geochemistry for camp-scale VMS exploration targeting in covered terranes.

**Highlights**

1. Integrated regional geophysical and till-geochemical datasets were used to predict concealed VMS mineralization in the Bathurst Mining Camp using machine learning
2. Geologically Constrained Class Labels were used to improve the representation of non-mineralized conditions in machine learning training data.
3. Th/K emerged as the most influential predictor, highlighting the importance of hydrothermal alteration signatures for VMS prospectivity mapping.

**Keywords:** mineral prospectivity mapping; volcanogenic massive sulphide; Bathurst Mining Camp; till geochemistry; class negative label; geophysical derivatives

# **1\. Introduction**

Volcanogenic massive sulphide (VMS) deposits are major global repositories of base metals, including copper, zinc, and lead, as well as associated precious metals such as gold and silver. Several commodities commonly associated with VMS systems, including zinc, copper, indium, bismuth, tin, and antimony, are classified as critical minerals that are essential for green-energy transition technologies and global decarbonization efforts (Franklin et al., 2005; Galley et al., 2007). Historically, VMS discovery relied on identifying shallow, outcropping mineralization. However, most near-surface deposits within mature exploration districts have already been identified, forcing exploration programs to target concealed systems beneath glacial overburden and transported cover (Goodfellow and McCutcheon, 2003). In glaciated terrains, successful exploration increasingly depends on the integration of regional geophysical and geochemical datasets capable of detecting indirect signatures of mineralization.

The Bathurst Mining Camp (BMC) of northern New Brunswick, Canada, is one of the world's premier VMS districts, hosting more than 45 known deposits and representing a major centre of historical base-metal production (Goodfellow, 2007; Goodfellow & McCutcheon, 2003). In addition to its historical significance, the camp contains commodities such as zinc, copper, indium, bismuth, tin, and antimony that appear on Canada's 2024 Critical Minerals List (Natural Resources Canada, 2024). The camp is hosted within the Cambro-Ordovician Tetagouche Group and is characterized by structurally complex bimodal volcanic-sedimentary sequences that have undergone intense polyphase deformation and ductile shear-zone development (van Staal et al., 2003). This structural complexity, combined with variable thicknesses of glacial till, obscures the surface expression of mineralized zones and complicates exploration targeting. Despite decades of exploration and the availability of extensive geoscientific datasets, relatively few new discoveries have been made in recent years, while an estimated 70% of the camp's prospective ground remains untested beneath glacial cover or insufficiently explored (McCutcheon & Walker, 2020). This contrast between data abundance and exploration success highlights the need for improved approaches to regional-scale mineral prospectivity mapping.

Machine learning (ML) has emerged as a powerful framework for mineral prospectivity mapping (MPM) because it enables the integration of diverse geoscientific datasets and the identification of complex, non-linear relationships associated with mineralization (Carranza & Laborte, 2015; Rodriguez-Galiano et al., 2014; Zuo et al., 2019). Within the BMC, Parsa et al. (2023) demonstrated the effectiveness of ML-based prospectivity mapping for volcanogenic-hosted massive sulphide deposits using airborne magnetic, radiometric, electromagnetic, and till-geochemistry datasets from the EXTECH II program. However, that study was restricted to the northeastern Brunswick Belt and incorporated only three till-geochemistry elements (Pb, Zn, and Cu). Moreover, Parsa et al. (2023) reported that airborne electromagnetic data commonly yielded unreliable predictors because of the masking effects associated with conductive Carboniferous overburden. Consequently, no study has yet evaluated the predictive value of integrating the complete airborne geophysical compilation of Ugalde et al. (2018) with the full New Brunswick till-geochemistry database at the scale of the entire BMC. This represents a significant knowledge gap because the provincial till-geochemistry dataset captures a much broader spectrum of geological information, including hydrothermal pathfinder elements and critical-mineral indicators such as indium, bismuth, tin, and antimony. Whether camp-scale integration of these complementary datasets can improve the prediction of concealed VMS mineralization therefore remains an important unresolved question.

A second challenge concerns the effective integration and interpretation of geophysical and geochemical information in regional VMS exploration. These datasets capture fundamentally different but complementary aspects of the mineral system. Geophysical surveys image lithological boundaries, structures, and physical-property contrasts that influence hydrothermal fluid flow and ore deposition, whereas geochemical datasets record elemental dispersion patterns and alteration footprints associated with mineralizing processes. The effectiveness of ML-based prospectivity mapping therefore depends not only on the quantity of available data but also on the physical and statistical integrity of the predictor variables (Zuo et al., 2021). For geophysical datasets, structural information is commonly enhanced through derivative products that emphasize geological boundaries and potential fluid pathways. However, derivative calculations performed on reprojected or interpolated grids may introduce spatial artifacts and distort high-frequency geological signals (Thomas et al., 2000). Similarly, radiometric datasets provide valuable information on hydrothermal alteration through radioelement distributions and ratios associated with potassic and sericitic alteration around submarine vent systems (Shives et al., 1997).

Geochemical datasets introduce additional complexity because elemental concentrations are compositional in nature and are subject to the constant-sum constraint (Aitchison, 1986; Filzmoser et al., 2009). Under these conditions, conventional multivariate analyses can generate spurious relationships that reflect data closure rather than genuine geological processes. Compositional data analysis (CoDA) provides a rigorous framework for addressing these limitations (Egozcue et al., 2003). Nevertheless, the question of how best to reconcile broad multivariate geochemical signatures that characterize regional hydrothermal systems with localized elemental anomalies that may provide direct evidence of mineralization remains. This challenge is particularly important for till-geochemistry datasets, where regional trends and localized pathfinder signatures may contain complementary information about concealed VMS systems (Parkhill & Doiron, 2003).

A further challenge in data-driven MPM is the negative-label for mineral absence problem. Mineral prospectivity mapping is typically formulated as a binary classification task requiring both positive and negative training labels (Parsa & Cumani, 2025). While positive labels can be assigned to known mineral occurrences, identifying reliable negative labels is considerably more difficult because the absence of a known deposit does not guarantee the absence of mineralization (Carranza & Laborte, 2015). Consequently, negative-label uncertainty can strongly influence model performance and predictive outcomes. Recent studies have shown that geologically informed negative-label strategies can improve model discrimination and exploration targeting by reducing ambiguity within the negative class and providing a more defensible representation of negative evidence (Barbet-Massin et al., 2012; Maepa et al., 2021; Parsa & Cumani, 2025).

Spatial autocorrelation presents an additional obstacle to the development of reliable ML-based prospectivity models. Nearby observations commonly share similar geological, geophysical, and geochemical characteristics, resulting in statistical dependence within spatial datasets. When observations are partitioned randomly into training and validation subsets, model performance may be artificially inflated because validation samples are not truly independent of the training data (Brenning, 2012; Roberts et al., 2017). Obtaining realistic estimates of model performance therefore requires validation approaches that account for spatial dependence. Beyond predictive accuracy, exploration programs also require models that can efficiently prioritize targets for follow-up investigation. Consequently, measures of targeting efficiency are essential complements to conventional classification metrics and provide a more practical assessment of exploration value (Bonham-Carter, 1994; Carranza, 2008).

Against this background, we present a camp-scale machine-learning prospectivity analysis of the Bathurst Mining Camp, integrating the regional airborne geophysical compilation of Ugalde et al. (2018) with a unified 17-element New Brunswick till-geochemistry database. Specifically, this study addresses four key areas: (1) integration of regional geophysical and till geochemical datasets to support camp-scale prediction of concealed VMS mineralization; (2) identification of the geophysical and geochemical variables that contribute most strongly to prospectivity predictions; (3) application of geologically informed negative-label constraints to construct a more representative training dataset for machine learning-based prospectivity modelling; and (4) development of a data-driven framework for ranking and prioritizing exploration targets for follow-up investigation. By building on decades of geological, geochemical, and geophysical research in the Bathurst Mining Camp, we seek to enhance the targeting of concealed mineralization and advance mineral prospectivity mapping methodologies for mature, data-rich VMS districts elsewhere. This open, optimization-aware approach aligns with the Exploration Information System (EIS) framework proposed by Daviran et al. (2026), which advocates for systematic, reproducible mineral prospectivity architectures.

**2.0 Regional Geological Setting**

**2.1 Tectonic and Stratigraphic Framework**

The Bathurst Mining Camp (BMC) occupies the Gander Zone of the northern Appalachian Orogen in New Brunswick, Canada (Fig. 1; Rogers & van Staal, 2003; van Staal et al., 2003). Its geological architecture reflects the evolution of the Cambro-Ordovician Tetagouche-Four Falls back-arc basin, which formed during the rifting of the Popelogan arc from the Gondwanan passive margin. Subsequent Taconic, Salinic, and Acadian orogenic events progressively closed the basin and tectonically imbricated the volcanic and sedimentary successions into a series of thrust-bounded structural blocks (Goodfellow & McCutcheon, 2003; van Staal et al., 2003).

The geological map (Fig. 1) highlights the dominance of Middle Ordovician volcanic and sedimentary assemblages, particularly the Tetagouche Group, which occupies much of the central region and hosts most known mineral occurrences, including the Brunswick No. 12 (B12) and Brunswick No. 6 (B6) deposits. The Tetagouche Group comprises the Nepisiguit Falls Formation (felsic volcaniclastics and tuffs), the Flat Landing Brook Formation (rhyolite flows and hyaloclastites), and the Boucher Brook Formation (tholeiitic pillow basalts, black shales, and iron formation) (Goodfellow, 2007). The California Lake Group forms an extensive belt along the northern part of the camp and hosts additional VMS deposits, including Caribou and Restigouche.

The para-autochthonous Miramichi Group forms the basement succession and consists mainly of quartzarenites and carbonaceous argillites deposited on the Gondwanan passive margin prior to rifting. Other mapped units include the Fournier Group, Sheephouse Brook Group, Upsalquitch Gabbro, granitic intrusions, and younger Silurian-Carboniferous volcanic and sedimentary rocks. The distribution of lithological units and the abundance of thrust faults shown on Figure 1 emphasize the strong structural modification of the original basin architecture and the importance of tectonic juxtaposition in preserving and exposing mineralized stratigraphy.

**2.2 VMS Deposit Style and Hydrothermal Alteration**
Deposits within the BMC belong predominantly to the bimodal-siliciclastic volcanogenic massive sulphide (VMS) subtype (Galley et al., 2007; Franklin et al., 2005). Mineralization is concentrated within the Tetagouche and California Lake groups and is commonly localized near felsic volcanic-sedimentary contacts. Typical deposits comprise a chlorite-silica-pyrite stockwork developed within a sub-seafloor hydrothermal feeder system, overlain by a stratiform massive sulphide lens dominated by pyrite, sphalerite, and galena, which hosts the bulk of the Zn-Pb-Ag-Au resource. Distal hydrothermal plume activity is commonly represented by jasperous or magnetite-rich iron formations (Goodfellow, 2007).

Hydrothermal alteration forms extensive halos surrounding mineralization, progressing outward from a proximal quartz-chlorite-pyrite assemblage to sericite-carbonate-pyrite alteration. These alteration zones are important exploration vectors and are commonly associated with potassium enrichment and thorium depletion detectable in airborne radiometric datasets (Shives et al., 1997; Goodfellow, 2007).

**2.3 Structural Controls and Exploration Implications**
The study area is characterized by a dense network of thrust faults and subsidiary brittle structures that dissect the Ordovician volcanic belts (Fig. 1). Multiple phases of Appalachian deformation under greenschist-facies conditions have significantly modified the original geometry of the VMS systems (van Staal et al., 2003; Rogers and van Staal, 2003). Early thrusting and nappe emplacement structurally repeated favourable host sequences, while later folding and faulting further fragmented and redistributed mineralized horizons. The concentration of known mineral occurrences along major structural corridors indicates that both ore preservation and present-day exposure are strongly controlled by these tectonic processes.

The polyphase deformation history eliminates simple surface expression of mineralization, elevates cover thickness through repeated structural stacking, and makes geophysical imaging of shear zones and density contrasts an indispensable complement to geochemical sampling (Parkhill & Doiron, 2003; Thomas et al., 2000). These characteristics make integrated structural, geophysical, and geochemical approaches essential for targeting concealed VMS deposits within the Bathurst Mining Camp.

**3. Methodology**

Geophysical, geological, drill-hole, and till geochemistry datasets were obtained from the New Brunswick Department of Natural Resources (NBDNR) through the department's ArcGIS REST services and integrated into QGIS 4.2.0 (Belém do Pará; QGIS Development Team, 2026) for visualization, quality control, and spatial data management.

The prospectivity mapping framework was structured as a multi-stage ML pipeline progressing from raw data compilation and grid-derivative computation to spatial machine learning and area-normalized validation (Fig. 2). The workflow was implemented using custom Python scripts and comprise of five key components: (1) data compilation and native-grid preprocessing; (2) compositional geochemical analysis; (3) feature extraction and engineering; (4) spatial block cross-validation and model training; and (5) full-extent mapping and interpretability.

## **3.1 Geophysical Datasets and Derivative Computation**

Airborne geophysical grids over the BMC were used, including Total Magnetic Intensity (RTMI), Bouguer gravity, and gamma-ray spectrometric (radiometric) grids for uranium, thorium, and potassium (Figs. 3a-e). All input grids are based on the New Brunswick Stereographic Double (NAD83; EPSG:2953).

To preserve structural boundaries and prevent grid distortions caused by spatial resampling, horizontal and vertical derivatives were computed in the Fourier domain on the original survey grids prior to cell-size transformation to 100 m (Blakely, 1995):

<<<<<<< HEAD
**First Vertical Derivative (FVD):** Computed via multiplication by the radial wavenumber $|\mathbf{k}|$ in the two-dimensional Fourier domain, followed by the inverse transform, to enhance high-frequency near-surface structural and lithological contacts (Blakely, 1995):

$$\text{FVD}(\mathbf{r}) = \mathcal{F}^{-1}\!\left\{ |\mathbf{k}|\, F(\mathbf{k}) \right\} \tag{1}$$

where $F(\mathbf{k})$ is the two-dimensional Fourier transform of the potential-field grid $f(\mathbf{r})$, $|\mathbf{k}| = \sqrt{k_x^2 + k_y^2}$ is the radial wavenumber, and $k_x, k_y$ are the horizontal wavenumbers.

**Total Horizontal Gradient (THG):** Derived as the magnitude of the horizontal gradient vector, highlighting density and susceptibility contrasts at geological boundaries (Verduzco et al., 2004):

$$\text{THG}(\mathbf{r}) = \sqrt{\left(\frac{\partial f}{\partial x}\right)^{\!2} + \left(\frac{\partial f}{\partial y}\right)^{\!2}} \tag{2}$$

where the spatial partial derivatives $\partial f / \partial x$ and $\partial f / \partial y$ are evaluated via their Fourier-domain operators $\mathcal{F}^{-1}\{i k_x F(\mathbf{k})\}$ and $\mathcal{F}^{-1}\{i k_y F(\mathbf{k})\}$, respectively.

**Tilt Derivative (TDR):** Calculated as the arctangent of the ratio of the FVD to the THG, equalizing amplitude variations between shallow and deep structural sources and providing robust edge-detection filters that delineate fault geometries and volcanic contacts (Miller & Singh, 1994):

$$\text{TDR}(\mathbf{r}) = \arctan\!\left(\frac{\text{FVD}(\mathbf{r})}{\text{THG}(\mathbf{r})}\right) \tag{3}$$
=======
**First Vertical Derivative (FVD):** Computed via multiplication by the radial wavenumber |_k_| in the two-dimensional Fourier domain, followed by the inverse transform, to enhance high-frequency near-surface structural and lithological contacts (Blakely, 1995):

(1)

where is the two-dimensional Fourier transform of the potential-field grid , and , are the horizontal wavenumbers.

**Total Horizontal Gradient (THG):** Derived as the magnitude of the horizontal gradient vector, highlighting density and susceptibility contrasts at geological boundaries (Verduzco et al., 2004):

(2)

where the partial derivatives are computed via the Fourier-domain equivalents , respectively.

**  
Tilt Derivative (TDR):** Calculated as the arctangent of the ratio of the FVD to the THG, equalizing amplitude variations between shallow and deep structural sources and providing robust edge-detection filters that delineate fault geometries and volcanic contacts (Miller & Singh, 1994):

(3)
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

Radiometric grids were preprocessed to generate radioelement ratios (K/Th, U/Th, Th/K) to map alteration zones characterized by potassic enrichment or thorium depletion indicative of VMS-related hydrothermal systems (Shives et al., 1997).

## **3.2 Geochemical Datasets**

Till-geochemistry point data (Figs. 3f-j) were compiled from 17 separate single-element databases (Ag, As, Ba, Bi, Cd, Co, Cu, Fe, In, Mn, Mo, Ni, Pb, Sb, Sn, Tl, Zn) covering the BMC. The 17-element geochemical dataset was selected to encompass both direct indicators of VMS mineralization and broader geochemical signatures reflecting hydrothermal alteration, metal transport, and depositional processes. In addition to the principal ore metals (Zn, Pb, Cu, and Ag), the dataset includes critical-mineral pathfinder elements (In, Bi, Sn, and Sb) and elements commonly associated with sulphide-rich hydrothermal systems (Fe, Co, Ni, As, Ba, Mo, Mn, and Tl). This expanded geochemical suite allows machine-learning models to evaluate complex multivariate relationships and identify predictive signatures beyond those represented by conventional VMS pathfinders alone. Because spatial coordinates varied slightly across individual survey datasets, sample points were aligned by rounding coordinates to the nearest meter, yielding a unified point-geochemistry database of 2,753 unique locations. Inverse distance weighting (IDW) interpolation technique was used to generate geochemical surfaces (Shepard, 1968; Cardoso-Fernandes et al., 2022; McClenaghan et al., 2023):

<<<<<<< HEAD
$$\hat{z}(\mathbf{s}_0) = \frac{\sum_{i=1}^n w_i z(\mathbf{s}_i)}{\sum_{i=1}^n w_i}, \quad \text{where } w_i = d(\mathbf{s}_0, \mathbf{s}_i)^{-p} \tag{4}$$

where $\hat{z}(\mathbf{s}_0)$ is the predicted concentration at location $\mathbf{s}_0$, $z(\mathbf{s}_i)$ is the measured concentration at sample location $\mathbf{s}_i$, $d(\mathbf{s}_0, \mathbf{s}_i)$ is the Euclidean distance between locations, $p = 2$ is the weighting power parameter, and $n = 12$ is the number of nearest neighbors considered.
=======
(4)

where ẑ(s<sub>0</sub>) is the predicted concentration at location s<sub>0</sub>, z(s<sub>i</sub>) is the measured concentration at sample s<sub>i</sub>, d(s<sub>0</sub>, s<sub>i</sub>) is the Euclidean distance between locations, p = 2 is the power parameter, and n = 12 is the number of nearest neighbors considered.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

## **3.3 Compositional Geochemical Analysis**

To address the closed nature of compositional geochemical data, concentration values were transformed using the centered-log ratio (CLR) transformation. The CLR projects variables from the constrained simplex space into unbounded real space relative to the geometric mean of the composition (Aitchison, 1986; Egozcue et al., 2003; Filzmoser et al., 2018):

<<<<<<< HEAD
$$\text{clr}(\mathbf{x}) = \left[ \ln\left(\frac{x_1}{g(\mathbf{x})}\right), \ln\left(\frac{x_2}{g(\mathbf{x})}\right), \dots, \ln\left(\frac{x_D}{g(\mathbf{x})}\right) \right] \tag{5}$$

where $g(\mathbf{x}) = \left(\prod_{j=1}^D x_j\right)^{1/D}$ is the geometric mean of the $D$ geochemical elements. Compositional PCA and compositional FA with varimax rotation were applied to the CLR-transformed IDW surfaces to extract orthogonal multi-element associations representing primary lithological units and hydrothermal alteration footprints (Filzmoser et al., 2009). PCA was used to summarize dominant geochemical variance into a reduced set of orthogonal components, whereas FA was used to identify latent multi-element associations potentially related to lithological and hydrothermal processes. The use of both methods provides complementary representations of regional geochemical variability for machine-learning analysis.
=======
(5)

where is the geometric mean of the D geochemical elements. Compositional PCA and compositional FA with varimax rotation were applied to the CLR-transformed IDW surfaces to extract orthogonal multi-element associations representing primary lithological units and hydrothermal alteration footprints (Filzmoser et al., 2009). PCA was used to summarize dominant geochemical variance into a reduced set of orthogonal components, whereas FA was used to identify latent multi-element associations potentially related to lithological and hydrothermal processes. The use of both methods provides complementary representations of regional geochemical variability for machine-learning analysis.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

## **3.4 Feature Extraction and Engineering**

### **3.4.1 Spatial Labels and Negative-Label Representativeness**

The training dataset was constructed from two primary label groups. Positive labels (Y=1) comprised 45 known VMS occurrences within the BMC (van Staal et al., 2003). The generation of reliable negative labels is a fundamental challenge in mineral prospectivity mapping because the absence of a known deposit does not guarantee the absence of mineralization, particularly in incompletely explored regions (Carranza and Laborte, 2015). Although studies adapted from species distribution modelling frequently describe background training samples as pseudo-absences (Barbet-Massin et al., 2012), we adopt the term negative labels (Y = 0) to remain consistent with binary supervised classification terminology and to avoid asserting unverified geological absence. To reduce uncertainty in negative-label selection, the negative class was assembled using a hybrid class-label strategy that combined two distinct sources of evidence:

1. _Confirmed barren drill holes (n = 125):_ Exploration drill intercepts compiled from New Brunswick Geological Survey records that did not intersect economic VMS mineralization. These samples provide geologically verified barren environments and anchor the negative class to locations with documented exploration results (Nykänen et al., 2008).
<<<<<<< HEAD
2. _Feature-space dissimilar negative labels (n = 125):_ Candidate locations were generated on a dense 100 m grid across the active geophysical survey footprint and ranked according to their Mahalanobis distance from the centroid of the positive class in multidimensional geophysical and geochemical feature space (Mahalanobis, 1936; Carranza, 2008). Mahalanobis distance was selected in preference to unweighted metrics such as Euclidean distance because it accounts for the covariance structure of the positive class, thereby reducing the influence of correlated predictor variables and providing a more robust measure of feature-space dissimilarity. This approach operationalizes the feature-space dissimilarity framework proposed by Parsa and Cumani (2025), that negative labels which are selected to be maximally dissimilar to known deposits in predictor space rather than simply distant in geographic space improve classifier discrimination and exploration-targeting efficiency. The Mahalanobis distance from each candidate point $\mathbf{c}$ to the deposit centroid $\boldsymbol{\mu}_+$ in standardised feature space is:

$$D_M(\mathbf{c}) = \sqrt{(\mathbf{c} - \boldsymbol{\mu}_+)^\top \boldsymbol{\Sigma}_+^{-1} (\mathbf{c} - \boldsymbol{\mu}_+)} \tag{6}$$

where $\boldsymbol{\Sigma}_+^{-1}$ is the inverse covariance matrix of the standardised positive-class feature vectors. Candidates are ranked in descending order of $D_M$; the 125 most dissimilar points are selected via stratified spatial sampling across four geographic quadrants to ensure geological dissimilarity is achieved without spatial clustering in any single zone of the survey footprint. A secondary minimum geographic guard distance of $d = 1000\text{ m}$ from any known deposit is retained as a hard constraint to prevent labelling points at the immediate margins of deposit footprints, but this geographic constraint is secondary and subordinate to the feature-space dissimilarity criterion. A fixed random seed (seed = 42) was applied to ensure reproducibility (Roberts et al., 2017).

The 1:1 ratio between confirmed barren drill intercepts ($n = 125$) and Mahalanobis-dissimilar pseudo-absences ($n = 125$) was selected based on three criteria: (1) to balance geologically ground-truthed evidence from direct subsurface drilling with statistical feature-space coverage of unexplored ground; (2) to maintain an overall positive-to-negative training ratio of approximately 1:5.56, consistent with empirical findings in data-driven prospectivity modeling indicating that negative pools 3–7 times larger than the positive set optimize discriminator calibration without inducing majority-class dominance (Parsa & Cumani, 2025); and (3) because exactly 125 confirmed barren GeoNB drill collars fell within the active master BMC raster extent and cleared the 1,000 m deposit buffer constraint. The sensitivity of model performance to this design choice is evaluated in Table 6, where the barren:Mahalanobis split is varied from 0/250 to 250/0 alongside a pure random background control, confirming that the balanced 125/125 configuration represents an optimal operational equilibrium that prevents artificial separability inflation while retaining verified drill-hole ground truth.

Features were extracted at the 295 training locations by sampling all geophysical derivative rasters and IDW-interpolated geochemical surfaces (Table 1). To preserve localized geochemical anomalies, raw elemental concentrations were incorporated through a nearest-neighbour spatial join. For each labelled location, the closest till-geochemistry sample within a maximum search radius of 1,000 m was identified, and the corresponding elemental concentrations were appended directly to the predictor matrix.

**Table 1.** The labelled dataset and predictor feature matrix.
=======
2. _Feature-space dissimilar negative labels (n = 125):_ Candidate locations were generated on a dense 100 m grid across the active geophysical survey footprint and ranked according to their Mahalanobis distance from the centroid of the positive class in multidimensional geophysical and geochemical feature space (Mahalanobis, 1936; Carranza, 2008). Mahalanobis distance was selected in preference to unweighted metrics such as Euclidean distance because it accounts for the covariance structure of the positive class, thereby reducing the influence of correlated predictor variables and providing a more robust measure of feature-space dissimilarity. This approach operationalizes the feature-space dissimilarity framework proposed by Parsa and Cumani (2025), that negative labels which are selected to be maximally dissimilar to known deposits in predictor space rather than simply distant in geographic space improve classifier discrimination and exploration-targeting efficiency. The Mahalanobis distance from each candidate point c to the deposit centroid μ<sub>+</sub> in standardised feature space is:

(6)

Where is the inverse covariance matrix of the standardised positive-class feature vectors. Candidates are ranked in descending order of D<sub>M</sub>; the 125 most dissimilar points are selected via stratified spatial sampling across four geographic quadrants to ensure geological dissimilarity is achieved without spatial clustering in any single zone of the survey footprint. A secondary minimum geographic guard distance of d = 1000 m from any known deposit is retained as a hard constraint to prevent labelling points at the immediate margins of deposit footprints, but this geographic constraint is secondary and subordinate to the feature-space dissimilarity criterion. A fixed random seed (seed = 42) was applied to ensure reproducibility (Roberts et al., 2017).

Features were extracted at the 295 training locations by sampling all geophysical derivative rasters and IDW-interpolated geochemical surfaces (Table 1). To preserve localized geochemical anomalies, raw elemental concentrations were incorporated through a nearest-neighbour spatial join. For each labelled location, the closest till-geochemistry sample within a maximum search radius of 1,000 m was identified, and the corresponding elemental concentrations were appended directly to the predictor matrix.

**Table 1**: The labelled dataset and predictor feature matrix
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

| **Labelled Dataset**  |       | **Predictor Feature**                        |     |
| --------------------- | ----- | -------------------------------------------- | --- |
| Samples               | 295   | Raw geochemistry                             | 17  |
| ---                   | ---   | ---                                          | --- |
| Positive labels (VMS) | 45    | Log-transformed geochemistry                 | 17  |
| ---                   | ---   | ---                                          | --- |
| Negative labels       | 250   | IDW geochemistry (raw) + 4 PCA + 4FA Factors | 25  |
| ---                   | ---   | ---                                          | --- |
| Class imbalance ratio | 1:5.6 | IDW geochemistry (log)                       | 17  |
| ---                   | ---   | ---                                          | --- |
|                       |       | Radiometric ratios (K/Th, U/Th & Th/K)       | 3   |
| ---                   | ---   | ---                                          | --- |
|                       |       | Geophysical features                         | 18  |
| ---                   | ---   | ---                                          | --- |
|                       |       | Composite score (MEAS)                       | 1   |
| ---                   | ---   | ---                                          | --- |
|                       |       | Spatial/identity attributes                  | 3   |
| ---                   | ---   | ---                                          | --- |
|                       |       | Target                                       | 1   |
| ---                   | ---   | ---                                          | --- |
|                       |       | Total                                        | 102 |
| ---                   | ---   | ---                                          | --- |

### **3.4.2 Secondary Feature Engineering**

Secondary features were engineered to capture additional mineralization criteria:

Analytic Signal (AS): Computed for both magnetics and gravity as the total amplitude of the gradient vector to isolate anomaly centres regardless of magnetization or polarization direction (Roest et al., 1992; Pham et al., 2022):

<<<<<<< HEAD
$$\text{AS}(\mathbf{r}) = \sqrt{\left(\frac{\partial f}{\partial x}\right)^{\!2} + \left(\frac{\partial f}{\partial y}\right)^{\!2} + \left(\frac{\partial f}{\partial z}\right)^{\!2}} \tag{7}$$

Log-transformations: Applied to all raw and IDW-interpolated geochemical concentration columns to stabilize variance and normalize the right-skewed frequency distributions characteristic of trace-element geochemistry (Reimann et al., 2008):

$$x_{\text{log}} = \ln(x_i + 1) \tag{8}$$

where $x_i$ is the raw elemental concentration and the shift of $+1$ prevents undefined values at zero-concentration observations.

Multi-Element Anomaly Score (MEAS): A geologically weighted composite indicator was calculated to capture anomalous concentrations of VMS pathfinder elements following the general principles of multivariate geochemical anomaly analysis outlined by Carranza (2008). Because VMS mineralization is characterized by the co-occurrence of multiple pathfinder elements, the MEAS was used to represent their collective enrichment within a single predictor. The objective was to enhance the expression of hydrothermal geochemical signatures while reducing reliance on individual elemental anomalies. Pathfinder concentrations were transformed so that each feature has a mean of 0 and a variance of 1 and weighted ($w_i$) according to their diagnostic association with massive sulphide mineralization:

$$\text{MEAS} = \sum_{i=1}^p w_i \cdot \text{scale}(x_i) = \sum_{i=1}^p w_i \left( \frac{x_i - \bar{x}_i}{s_i} \right) \tag{9}$$

where MEAS is the multi-element anomaly score, $p$ is the number of selected pathfinder elements, $x_i$ is the concentration of pathfinder element $i$, $\text{scale}(x_i) = (x_i - \bar{x}_i)/s_i$ is the standardized value of $x_i$, and $w_i$ is the weight assigned to element $i$, with $\bar{x}_i$ and $s_i$ denoting the mean and standard deviation of element $i$, respectively.

Weights were assigned based on established paragenetic sequences and metallogeny of BMC deposits (Franklin et al., 2005; Goodfellow, 2007; McCutcheon et al., 2003): primary ore metals Zn and Pb ($w = 1.5$); proximal stockwork indicator Cu ($w = 1.2$); high-temperature hydrothermal indicators Mo and Ag ($w = 1.0$); sulphosalt and sphalerite associates Bi and Cd ($w = 0.8$); and distal volatile pathfinders Sb and As ($w = 0.5$). Weights reflect established diagnostic associations from prior geological literature rather than data-driven feature importance scores, avoiding circularity with the downstream machine-learning classifiers.

## **3.5 Data Quality Filtering and Class Balancing**

All 17 raw geochemical elements were retained as predictor variables, as none exceeded the 75% missing-data threshold at labelled sample locations. The sparsest variables, Bi (60.3%), In (60.3%), Tl (60.3%), and Mn (58.3%), were preserved as predictors and complemented by their spatially complete IDW-interpolated surfaces, which provided values for all 295 labelled locations (Table 1). Missing values in the retained sparse features were imputed with column-wise medians, calculated solely from the training folds to avoid data leakage. This approach is robust to non-normal distributions and is widely applied in geoscientific datasets containing sparse geochemical variables (Carranza & Laborte, 2015; Reimann et al., 2008). To verify that these sparse elements provided predictive utility rather than extraneous noise, a sensitivity analysis comparing models trained with versus without these four sparse raw columns — with IDW-interpolated surfaces retained in both configurations — was conducted for both Random Forest and XGBoost (Table 7; Supplementary Table S2). Both configurations produced comparable cross-validation performance, with ROC-AUC differences of 0.026 for both RF and XGBoost, well within the fold-to-fold standard deviation (±0.05–0.08), indicating that model performance is not critically sensitive to the inclusion of these sparse columns. Given that all four elements fall below the established 75% null threshold, that their IDW counterparts provide spatially continuous coverage at all labelled locations, and that the performance difference is not statistically meaningful under 5-fold spatial block cross-validation, all 17 raw geochemical elements were retained in the published 60-feature predictor matrix to preserve the full dual-scale geochemical representation.

The final training dataset comprised 45 positive labels representing known VMS deposits and 250 negative labels, resulting in a class ratio of approximately 1:5.6 (Table 1). This class imbalance risks skewing machine-learning models in favor of the majority class, limiting their capacity to accurately detect mineralized environments (Li et al., 2020). To address this, the Synthetic Minority Over-sampling Technique (SMOTE; Chawla et al., 2002; Nidhi et al., 2026) was used to oversample the minority (positive) class during training. SMOTE generates synthetic positive instances by interpolating between neighboring minority-class samples in feature space:

$$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{\text{neighbor}} - \mathbf{x}_i), \quad \lambda \sim U(0, 1) \tag{10}$$

where $\mathbf{x}_i$ is a minority-class sample, $\mathbf{x}_{\text{neighbor}}$ is one of its nearest minority neighbors, and $\lambda$ is a uniform random variable between 0 and 1. Application of SMOTE is geologically reasonable in VMS prospectivity mapping because mineralized systems are commonly associated with continuous physical and geochemical gradients expressed through hydrothermal alteration halos, pathfinder-element dispersion patterns, and geophysical anomaly responses. Interpolation between known mineralized samples therefore generates synthetic observations that occupy plausible intermediate regions of the prospectivity feature space rather than arbitrary locations.
=======
(7)

Log-transformations: Applied to all raw and IDW-interpolated geochemical concentration columns to stabilize variance and normalize the right-skewed frequency distributions characteristic of trace-element geochemistry (Reimann et al., 2008):

(8)

where xᵢ is the raw elemental concentration and the shift of +1 prevents undefined values at zero-concentration observations.

Multi-Element Anomaly Score (MEAS): A geologically weighted composite indicator was calculated to capture anomalous concentrations of VMS pathfinder elements following the general principles of multivariate geochemical anomaly analysis outlined by Carranza, (2008). Because VMS mineralization is characterized by the co-occurrence of multiple pathfinder elements, the MEAS was used to represent their collective enrichment within a single predictor. The objective was to enhance the expression of hydrothermal geochemical signatures while reducing reliance on individual elemental anomalies. Pathfinder concentrations were transformed so that each feature has a mean of 0 and a variance of 1 and weighted (wᵢ) according to their diagnostic association with massive sulphide mineralization.

(9)

where MEAS is the multi-element anomaly score, p is the number of selected pathfinder elements, x<sub>i</sub> is concentration of pathfinder element i, scale(x<sub>i</sub>) =((x<sub>i</sub>\-µ<sub>i</sub>)/σ<sub>i</sub>) is the standardized value of x<sub>i</sub> and w<sub>i</sub> is the weight assigned to element i; whereas µ<sub>i</sub> and σ<sub>i</sub> are the mean and standard deviation of element i respectively.

## **3.5 Data Quality Filtering and Class Balancing**

All 17 raw geochemical elements were retained as predictor variables, as none exceeded the 75% missing-data threshold at labelled sample locations. The sparsest variables, Bi (60.3%), In (60.3%), Tl (60.3%), and Mn (58.3%), were preserved as predictors and complemented by their spatially complete IDW-interpolated surfaces, which provided values for all 295 labelled locations (Table 1). Missing values in the retained sparse features were imputed with column-wise medians, calculated solely from the training folds to avoid data leakage. This approach is robust to non-normal distributions and is widely applied in geoscientific datasets containing sparse geochemical variables (Carranza & Laborte, 2015; Reimann et al., 2008). To verify that these sparse elements provided predictive utility rather than extraneous noise, a sensitivity analysis comparing models trained with versus without these four sparse raw columns — with IDW-interpolated surfaces retained in both configurations — was conducted for both Random Forest and XGBoost (Table 7; Supplementary Table S2). Both configurations produced comparable cross-validation performance, with ROC-AUC differences of 0.026 for both RF and XGBoost, well within the fold-to-fold standard deviation (±0.05–0.08), indicating that model performance is not critically sensitive to the inclusion of these sparse columns. Given that all four elements fall below the established 75% null threshold, that their IDW counterparts provide spatially continuous coverage at all labelled locations, and that the performance difference is not statistically meaningful under 5-fold spatial block cross-validation, all 17 raw geochemical elements were retained in the published 60-feature predictor matrix to preserve the full dual-scale geochemical representation.

The final training dataset comprised 45 positive labels representing known VMS deposits and 250 negative labels, resulting in a class ratio of approximately 1:5.6 (Table 1). This class imbalance risks skewing machine-learning models in favor of the majority class, limiting their capacity to accurately detect mineralized environments (Li et al., 2020). To address this, the Synthetic Minority Over-sampling Technique (SMOTE; Chawla et al., 2002; Nidhi et al., 2026) was used to oversample the minority (positive) class during training. SMOTE generates synthetic positive instances by interpolating between neighboring minority-class samples in feature space:

(10)

where x<sub>i</sub>​ is a minority-class sample, x<sub>neighbour</sub>​ is one of its nearest minority neighbors, and λ is a random value between 0 and 1. Application of SMOTE is geologically reasonable in VMS prospectivity mapping because mineralized systems are commonly associated with continuous physical and geochemical gradients expressed through hydrothermal alteration halos, pathfinder-element dispersion patterns, and geophysical anomaly responses. Interpolation between known mineralized samples therefore generates synthetic observations that occupy plausible intermediate regions of the prospectivity feature space rather than arbitrary locations.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

To prevent spatial data leakage, SMOTE was applied within the training subset of each spatial block cross-validation fold and not to the validation data. The minority class was augmented to match the size of the negative class (n = 250 per class), yielding a balanced training dataset of N = 500 samples. This procedure improved representation of mineralized environments during model training while ensuring that performance metrics were evaluated using geographically independent observations.

## **3.6 Spatial Block Cross-Validation**

To address spatial autocorrelation and prevent overly optimistic performance estimates arising from geographically proximate observations, a 5-fold spatial block cross-validation scheme was implemented (Brenning, 2012). The study area was divided into five distinct geographic blocks (folds) containing approximately equal numbers of points/samples using quantiles of sample easting coordinates (Fig. 4). In each iteration, one fold was held out for testing while the other four were used for training, maintaining geographic separation between training and validation data (Roberts et al., 2017). This provides a more realistic evaluation of model performance in unexplored regions. Class balancing was applied separately within each training fold, following spatial partitioning of the dataset. This ensured synthetic samples were derived from training data only, while keeping validation observations spatially independent and unaffected by the balancing process.

## **3.7 Classifiers and Hyperparameter Tuning**

<<<<<<< HEAD
Random Forest (RF; Breiman, 2001; Rodriguez-Galiano et al., 2014; Sun, et al. 2019; Mami Khalifani, 2025) and Extreme Gradient Boosting (XGBoost; Chen & Guestrin, 2016; Parsa, 2021; Ghane et al., 2026) classifiers were trained and optimized using automated Bayesian hyperparameter search within the spatial cross-validation framework. The importance of automated hyperparameter optimization in mineral prospectivity modeling has been demonstrated in recent studies across both supervised algorithms (Daviran et al., 2021, 2025) and unsupervised frameworks (Daviran & Maghsoudi, 2026). In this study, hyperparameter optimization was executed using Optuna (Akiba et al., 2019) across 50 trials per algorithm, using the mean spatial block CV ROC-AUC as the objective function.

RF was implemented using the Gini impurity criterion (Breiman, 2001) to recursively partition the predictor space, whereas XGBoost was trained using a binary logistic loss and gradient boosting optimization (Chen & Guestrin, 2016). To prevent overfitting given the limited number of known VMS occurrences ($n = 45$, corresponding to ~30–35 training positives per fold), hyperparameter search spaces were bounded conservatively: tree depth was constrained, feature subsampling fractions were enforced, minimum leaf samples were required, and explicit $L_1$ and $L_2$ regularization penalties were incorporated in XGBoost (Table 2). Class weights were balanced in both classifiers to further mitigate residual effects of class imbalance.

**Table 2.** Hyperparameter search spaces and optimal values identified via 50-trial Optuna Bayesian optimization.

| Algorithm | Hyperparameter | Search Space / Distribution | Best Value | Overfitting Mitigation Rationale |
|---|---|---|---|---|
| **Random Forest** | `n_estimators` | Int [100, 800] | **652** | High ensemble averaging stabilizes spatial variance |
| | `max_depth` | Int [3, 30] | **30** | Constrains extreme tree complexity |
| | `min_samples_leaf` | Int [1, 20] | **1** | Enforces minimum sample support at leaf nodes |
| | `max_features` | Categorical ['sqrt', 'log2', 0.3, 0.5] | **'log2'** | Random feature subsampling decorrelates individual trees |
| **XGBoost** | `n_estimators` | Int [100, 500] | **229** | Bounded iterations prevent boosting overfit on minority class |
| | `max_depth` | Int [3, 10] | **5** | Shallow depth limits complex high-order interaction learning |
| | `learning_rate` | Log-uniform [0.01, 0.3] | **0.0566** | Conservative shrinkage factor regularizes gradient steps |
| | `subsample` | Uniform [0.5, 1.0] | **0.5480** | Stochastic row subsampling prevents local cluster overfitting |
| | `colsample_bytree` | Uniform [0.5, 1.0] | **0.9113** | Column subsampling reduces reliance on dominant features |
| | `reg_alpha` ($L_1$) | Log-uniform [$10^{-5}$, 1.0] | **0.0018** | Lasso penalty induces sparsity in feature weights |
| | `reg_lambda` ($L_2$) | Log-uniform [$10^{-5}$, 10.0] | **$1.06 \times 10^{-6}$** | Ridge penalty shrinks leaf weights against extreme predictions |
=======
Random Forest (RF; Breiman, 2001; Rodriguez-Galiano et al., 2014; Sun, et al. 2019; Mami Khalifani, 2025) and Extreme Gradient Boosting (XGBoost; Chen & Guestrin, 2016; Parsa, 2021; Ghane et al., 2026) classifiers were trained and optimized using randomized hyperparameter search within the spatial cross-validation framework. RF was implemented using the Gini impurity criterion (Breiman, 2001) to recursively partition the predictor space into homogeneous classes, whereas XGBoost was trained using a binary logistic objective function and gradient boosting optimization (Chen & Guestrin, 2016). Hyperparameter optimization was performed to identify model configurations that maximized predictive performance while reducing the risk of overfitting. Class weights were balanced in both classifiers to further mitigate residual effects of class imbalance.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

## **3.8 Performance Metrics Evaluation**

Model performance was assessed using four complementary metrics: Receiver Operating Characteristic Area Under the Curve (ROC<sub>AUC</sub>), Average Precision (AP), Balanced Accuracy (BA), and Success Rate Area Under the Curve (SR<sub>AUC</sub>) (Davis & Goadrich, 2006; Fawcett, 2006; Nykänen et al., 2015; Sun, et al. 2019; Parsa & Carranza, 2021; Nidhi et al., 2026).

### **3.8.1 Receiver Operating Characteristics (ROC<sub>AUC</sub>)**

<<<<<<< HEAD
ROC<sub>AUC</sub> evaluates the ability of a classifier to discriminate between mineralized and non-mineralized locations across all probability thresholds by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR):

$$\text{ROC}_{\text{AUC}} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR}) \tag{11}$$

where TPR(FPR) represents the ROC curve operating characteristic. As a threshold-independent metric, ROC<sub>AUC</sub> measures discriminatory power across the full operating range of the classifier, making it insensitive to any particular decision boundary (Fawcett, 2006) and is widely used in mineral prospectivity mapping studies.
=======
ROC<sub>AUC</sub> evaluates the ability of a classifier to discriminate between mineralized and non-mineralized locations across all probability thresholds by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR). It is given by:)

(11)

where TPR(FPR) is the ROC curve. As a threshold-independent metric, ROC<sub>AUC</sub> measures discriminatory power across the full operating range of the classifier, making it insensitive to any particular decision boundary (Fawcett, 2006) and is widely used in mineral prospectivity mapping studies.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

### **3.8.2 Average Precision (AP)**

Average Precision evaluates the precision-recall trade-off and is particularly informative for imbalanced datasets where positive observations represent only a small fraction of the total sample population:

<<<<<<< HEAD
$$\text{AP} = \sum_{k=1}^K (R_k - R_{k-1}) P_k \tag{12}$$

where $P_k$ and $R_k$ represent precision and recall at the $k$-th classification threshold across $K$ operating points (Davis & Goadrich, 2006). AP penalizes models that produce excessive false positives at high recall, emphasizing the ability of a model to recover positive samples while minimizing false-positive predictions.

### **3.8.3 Balanced Accuracy (BA)**

BA provides a useful threshold-specific measure of performance for balanced training datasets generated through SMOTE. It equally weighs sensitivity and specificity of the model's ability to correctly identify both VMS deposits and barren locations and is insensitive to class imbalance (Brodersen et al., 2010):

$$\text{BA} = \frac{1}{2} \left( \text{Sensitivity} + \text{Specificity} \right) = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right) \tag{13}$$

where TP, TN, FP, and FN are true positives, true negatives, false positives, and false negatives, respectively. BA will equal 0.5 for a no-skill classifier regardless of class frequencies. A BA close to 0.7 or higher confirms that neither class dominates predictions at the operational decision threshold.

### **3.8.4 Success Rate (SR<sub>AUC</sub>)**

Evaluates targeting efficiency by plotting the cumulative fraction of known VMS deposits captured ($f_d$) against the cumulative fraction of total study area covered ($f_a$) when cells are ranked by prospectivity index in descending order (Carranza, 2008):

$$\text{SR}_{\text{AUC}} = \int_0^1 f_d(f_a) \, df_a \tag{14}$$

An SR<sub>AUC</sub> value of 0.5 indicates random targeting performance, whereas a value of 1.0 represents perfect ranking of mineralized locations. SR<sub>AUC</sub> directly quantifies the economic efficiency of the model for drill-targeting decisions by measuring how much of the deposit inventory is captured within a minimal search area.

## **3.9 Full-Extent Mapping and Model Interpretability**
=======
(12)

where P = Precision and R = Recall. AP penalizes models that produce excessive false positives at high recall (Davis & Goadrich, 2006). It emphasizes the ability of a model to recover positive samples while minimizing false-positive predictions.

### **3.8.3 Balanced Accuracy (BA)**

BA provides a useful threshold-specific measure of performance for balanced training datasets generated through SMOTE. It equally weighs sensitivity and specificity of the model's ability to correctly identify both VMS deposits and barren locations and is insensitive to class imbalance. Unlike standard accuracy, it is insensitive to class imbalance and will equal 0.5 for a no-skill classifier regardless of class frequencies (Brodersen et al., 2010). Balanced Accuracy was calculated as:

(13)

BA is particularly informative here because SMOTE-balanced training sets could in principle create a model that over-predicts the positive class; a BA close to 0.7 confirms that neither class dominates the predictions at the default threshold.

### **3.8.4 Success Rate (SR<sub>AUC</sub>)**

Evaluates targeting efficiency by plotting the cumulative fraction of known VMS deposits captured (f<sub>d</sub>) against the cumulative fraction of total study area covered (f<sub>a</sub>) when cells are ranked by prospectivity index in descending order (Carranza, 2008). It is calculated by:

(14)

An SR<sub>AUC</sub> value of 0.5 indicates random targeting performance, whereas a value of 1.0 represents perfect ranking of mineralized locations. SR<sub>AUC</sub> directly quantifies the economic efficiency of the model for drill-targeting decisions by measuring how much of the deposit inventory is captured within a minimal search area.

**3.9 Full-Extent Mapping and Model Interpretability**
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

Following model training and spatial cross-validation, both the Random Forest (RF) and XGBoost classifiers were projected across the full Bathurst Mining Camp prediction grid (953 × 1,253 cells; 1,194,109 total cells) to generate continuous prospectivity surfaces at 100 m spatial resolution. Prospectivity values represent the predicted probability of VMS mineralization for each grid cell, producing camp-scale maps suitable for comparison of spatial prediction patterns and delineation of prospective target areas. The resulting RF and XGBoost prospectivity maps were subsequently compared, and the classifier demonstrating the strongest performance under spatial cross-validation was recommended as the preferred model for exploration targeting.

To enhance model transparency and facilitate geological interpretation of model predictions, SHapley Additive exPlanations (SHAP; Lundberg & Lee, 2017; Ghane et al., 2026; Luo et al., 2026; Nidhi et al., 2026) were applied to both RF and XGBoost models. SHAP quantifies the contribution of individual predictors to model outputs by assigning feature-attribution values based on their marginal contribution across all possible feature combinations. Predictor importance was assessed using mean absolute SHAP values computed in probability space, enabling direct comparison of predictor influence between the two classifiers. The resulting SHAP values were used to identify the geophysical, radiometric, and geochemical variables that exert the strongest control on modelled VMS prospectivity and to support geological interpretation of the underlying mineral system.

# **4\. Results**

Results are presented for the compositional geochemical analyses, machine-learning model performance, feature-importance evaluation, and camp-scale prospectivity mapping. Comparative performance of the Random Forest and XGBoost classifiers is assessed using spatial block cross-validation metrics, followed by interpretation of the resulting prospectivity predictions.

<<<<<<< HEAD
## **4.1 Multivariate Geochemical Association**

Principal Component Analysis (PCA) and Factor Analysis (FA) of the CLR-transformed till geochemistry dataset identified four major geochemical associations that capture distinct patterns of elemental covariance (Table 3). The dominant association, represented by PC1/FA1, is characterized by elevated Zn, Pb, Co, Ni, Sb, Cu, Ba, and Fe, together with relative depletion of In and Mo. This component reflects the principal polymetallic VMS geochemical signature within the Bathurst Mining Camp and exhibited the strongest association with known VMS mineralization among the derived multivariate variables.
=======
### **4.1 Multivariate Geochemical Association**

Principal Component Analysis (PCA) and Factor Analysis (FA) of the CLR-transformed till geochemistry dataset identified four major geochemical associations that capture distinct patterns of elemental covariance (Table 2). The dominant association, represented by PC1/FA1, is characterized by elevated Zn, Pb, Co, Ni, Sb, Cu, Ba, and Fe, together with relative depletion of In and Mo. This component reflects the principal polymetallic VMS geochemical signature within the Bathurst Mining Camp and exhibited the strongest association with known VMS mineralization among the derived multivariate variables.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

The second association (PC2/FA2) is dominated by Bi and Cd enrichment and contrasts with lower In and Mo concentrations, defining a distinct geochemical population that is largely independent of the primary base-metal assemblage. PC3 expresses an Ag-enriched and As-depleted association, whereas PC4 is characterized by Sn enrichment coupled with Ag depletion. Similar patterns were identified through factor analysis, which produced broadly equivalent geochemical groupings following varimax rotation.

Factor Analysis further highlighted geochemically distinct associations within the dataset. FA3 contrasts Cu-As enrichment with Ba-Tl depletion, while FA4 is characterized by Bi enrichment and relative depletion of Ag, Mo, and Sn. Notably, FA4 subsequently emerged as one of the most influential composite geochemical predictors in the machine-learning models.

The PCA and FA results demonstrate that the till geochemistry dataset contains multiple orthogonal elemental associations representing distinct geochemical processes. These multivariate variables provide a compact representation of complex geochemical relationships and form important predictor inputs for subsequent VMS prospectivity modelling.

<<<<<<< HEAD
**Table 3.** Relationships and correlation strengths between the geochemical elements and the four principal components (PC) and factors (FA).
=======
**Table 2:** The relationships and correlation strengths between the geochemical elements and the four principal components (PC) and factors (FA)
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

| Element | Principal Components |        |        |        | Factors |        |        |        |
| ------- | -------------------- | ------ | ------ | ------ | ------- | ------ | ------ | ------ |
|         | PC1                  | PC2    | PC3    | PC4    | FA1     | FA2    | FA3    | FA4    |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Ag      | 0.098                | −0.085 | 0.664  | −0.528 | 0.339   | −0.043 | −0.101 | −0.362 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| As      | 0.251                | 0.095  | −0.360 | −0.142 | 0.637   | 0.449  | 0.311  | 0.237  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Ba      | 0.271                | −0.023 | 0.236  | 0.190  | 0.779   | 0.345  | −0.468 | 0.073  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Bi      | −0.002               | 0.580  | 0.032  | 0.186  | 0.321   | −0.603 | 0.257  | 0.513  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Cd      | −0.093               | 0.523  | −0.038 | −0.087 | 0.025   | −0.625 | 0.229  | 0.264  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Co      | 0.298                | −0.134 | −0.124 | −0.063 | 0.712   | 0.664  | −0.009 | −0.079 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Cu      | 0.274                | −0.060 | −0.340 | −0.184 | 0.642   | 0.641  | 0.364  | −0.023 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Fe      | 0.269                | −0.219 | −0.055 | −0.023 | 0.617   | 0.615  | −0.113 | −0.238 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| In      | −0.219               | −0.339 | −0.357 | −0.158 | −0.947  | 0.311  | −0.014 | 0.028  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Mn      | 0.292                | 0.167  | 0.022  | 0.045  | 0.923   | 0.224  | 0.111  | 0.043  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Mo      | −0.204               | −0.329 | 0.190  | 0.228  | −0.588  | −0.226 | −0.235 | −0.449 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Ni      | 0.285                | −0.017 | −0.086 | −0.107 | 0.776   | 0.452  | 0.219  | −0.149 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Pb      | 0.302                | −0.005 | 0.036  | 0.078  | 0.843   | 0.419  | −0.105 | 0.040  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Sb      | 0.279                | −0.108 | 0.076  | −0.141 | 0.728   | 0.459  | −0.057 | −0.159 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Sn      | 0.139                | −0.183 | 0.021  | 0.639  | 0.362   | 0.185  | −0.081 | −0.257 |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Tl      | 0.266                | 0.004  | 0.237  | 0.244  | 0.777   | 0.296  | −0.438 | 0.134  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |
| Zn      | 0.307                | 0.088  | −0.054 | −0.014 | 0.873   | 0.419  | 0.029  | 0.151  |
| ---     | ---                  | ---    | ---    | ---    | ---     | ---    | ---    | ---    |

<<<<<<< HEAD
## **4.2 Model Performance Under Spatial Cross-Validation**

Spatial block cross-validation (Fig. 4) results demonstrate strong predictive performance for both the Random Forest (RF) and XGBoost models (Table 4). RF achieved the highest overall discrimination and targeting performance, with a mean ROC<sub>AUC</sub> of 0.9318 ± 0.0368, Average Precision of 0.7245 ± 0.1476, and Success Rate of 0.9680, compared with 0.9098 ± 0.0369, 0.6226 ± 0.1481, and 0.9494, respectively, for XGBoost. XGBoost produced a marginally higher Balanced Accuracy (0.8456 ± 0.0654) than RF (0.8261 ± 0.0701).

Performance variability across the five spatial folds was low for both classifiers, with nearly identical ROC-AUC standard deviations (RF: ±0.0368; XGBoost: ±0.0369), indicating consistent predictive behaviour across geographically independent validation blocks. Overall, RF provided the strongest combination of discrimination and exploration-targeting performance (Fig. 5).

**Table 4.** Mean spatial block cross-validation performance metrics.
=======
### **4.2 Model Performance Under Spatial Cross-Validation**

Spatial block cross-validation (Fig. 4) results demonstrate strong predictive performance for both the Random Forest (RF) and XGBoost models (Table 3). RF achieved the highest overall discrimination and targeting performance, with a mean ROC<sub>AUC</sub> of 0.9318 ± 0.0368, Average Precision of 0.7245 ± 0.1476, and Success Rate of 0.9680, compared with 0.9098 ± 0.0369, 0.6226 ± 0.1481, and 0.9494, respectively, for XGBoost. XGBoost produced a marginally higher Balanced Accuracy (0.8456 ± 0.0654) than RF (0.8261 ± 0.0701).

Performance variability across the five spatial folds was low for both classifiers, with nearly identical ROC-AUC standard deviations (RF: ±0.0368; XGBoost: ±0.0369), indicating consistent predictive behaviour across geographically independent validation blocks. Overall, RF provided the strongest combination of discrimination and exploration-targeting performance (Fig. 5).

**Table 3.** Mean spatial block cross-validation performance metrics.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

| **Metric**                                                                        | **Random Forest** | **XGBoost**     |
| --------------------------------------------------------------------------------- | ----------------- | --------------- |
| Receiver Operating Characteristics, ROC<sub>AUC</sub> (mean ± standard deviation) | 0.9318 ± 0.0368   | 0.9098 ± 0.0369 |
| ---                                                                               | ---               | ---             |
| Average Precision (mean ± standard deviation)                                     | 0.7245 ± 0.1476   | 0.6226 ± 0.1481 |
| ---                                                                               | ---               | ---             |
| Balanced Accuracy (mean ± standard deviation)                                     | 0.8261 ± 0.0701   | 0.8456 ± 0.0654 |
| ---                                                                               | ---               | ---             |
| Success Rate, SR<sub>AUC</sub>                                                    | 0.968             | 0.949           |
| ---                                                                               | ---               | ---             |

<<<<<<< HEAD
## **4.3 VMS Prospectivity Patterns Across the Bathurst Mining Camp**
=======
**4.3 VMS Prospectivity Patterns Across the Bathurst Mining Camp**
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

The prospectivity maps generated by the Random Forest (RF) and XGBoost classifiers reveal broadly similar spatial patterns of predicted VMS prospectivity across the Bathurst Mining Camp (Fig. 6). In both models, elevated Prospectivity Index (PI) values form spatially coherent, elongate target corridors exhibiting a predominantly north-northeast to south-southwest (NNE-SSW) orientation, broadly parallel to the regional structural and stratigraphic framework of the Tetagouche Group. The highest-prospectivity zones form continuous linear belts that coincide with known VMS clusters and favourable volcanic horizons. The XGBoost model predicts a larger spatial footprint of high-prospectivity areas, whereas the RF model produces a more concentrated set of anomalies.

The RF prospectivity map exhibits a highly localized distribution of predicted mineralization potential, with PI values ranging from 0 to 1 and a median value of 0.049. High-priority targets (PI > 0.7) occupy 23,585 cells (2.0% of the study area), whereas moderate-to-high prospectivity zones (PI > 0.5) encompass 85,303 cells (7.1%). Very high-priority areas (PI > 0.9) are restricted to 2,678 cells (0.2%). In contrast, the XGBoost prospectivity map displays a lower median PI value (0.0025) and a more polarized probability distribution, delineating 59,588 cells (5.0%) exceeding PI > 0.7 and 92,200 cells (7.7%) exceeding PI > 0.5.

<<<<<<< HEAD
Despite differences in the extent and distribution of high-prospectivity areas, both classifiers identify several common prospective corridors associated with favourable volcanic stratigraphy and regional structural trends. Several anomalies with PI values exceeding 0.7 occur beyond the footprint of currently documented VMS deposits, highlighting additional prospective areas within covered portions of the camp. Since the RF map delineates a more spatially focused set of anomalies, it was therefore selected as the preferred prospectivity model based on its superior performance under spatial cross-validation (Table 4).

## **4.4 Key Predictors of VMS Prospectivity**

Mean absolute SHAP values, computed in probability space, were used to quantify predictor importance for the Random Forest (RF) and XGBoost models (Table 5). Both classifiers showed strong agreement regarding the primary controls on VMS prospectivity. The radiometric Th/K alteration ratio emerged as the highest-ranked predictor in both RF (0.0501) and XGBoost (0.0857), highlighting the importance of alteration-related radiometric signatures. Molybdenum-related variables were consistently among the most influential predictors, with IDW-interpolated molybdenum ranking third in RF and second in XGBoost. Zinc-related variables also featured prominently in both models, further emphasizing their importance as VMS pathfinder indicators.

Although the highest-ranked predictors were broadly consistent, the two models differed in their relative emphasis on specific data domains. RF assigned greater importance to radiometric variables and interpolated geochemical surfaces, including Zn, Bi, and Pb anomaly layers, whereas XGBoost relied on a broader combination of geochemical, radiometric, gravity, and magnetic predictors. Notably, upward-continued Bouguer gravity, magnetic analytic signal, and the radiometric U/Th ratio ranked among the most influential variables in XGBoost but were comparatively less important in RF.

**Table 5.** Top 10 predictor features ranked by mean absolute SHAP values for the Random Forest and XGBoost models. SHAP values were calculated in probability space and reflect each predictor's average contribution to modelled VMS prospectivity.
=======
Despite differences in the extent and distribution of high-prospectivity areas, both classifiers identify several common prospective corridors associated with favourable volcanic stratigraphy and regional structural trends. Several anomalies with PI values exceeding 0.7 occur beyond the footprint of currently documented VMS deposits, highlighting additional prospective areas within covered portions of the camp. Since the RF map delineates a more spatially focused set of anomalies, it was therefore selected as the preferred prospectivity model based on its superior performance under spatial cross-validation (Table 3).

### **4.4 Key Predictors of VMS Prospectivity**

Mean absolute SHAP values, computed in probability space, were used to quantify predictor importance for the Random Forest (RF) and XGBoost models (Table 4). Both classifiers showed strong agreement regarding the primary controls on VMS prospectivity. The radiometric Th/K alteration ratio emerged as the highest-ranked predictor in both RF (0.0501) and XGBoost (0.0857), highlighting the importance of alteration-related radiometric signatures. Molybdenum-related variables were consistently among the most influential predictors, with IDW-interpolated molybdenum ranking third in RF and second in XGBoost. Zinc-related variables also featured prominently in both models, further emphasizing their importance as VMS pathfinder indicators.

Although the highest-ranked predictors were broadly consistent, the two models differed in their relative emphasis on specific data domains. RF assigned greater importance to radiometric variables and interpolated geochemical surfaces, including Zn, Bi, and Pb anomaly layers, whereas XGBoost relied on a broader combination of geochemical, radiometric, gravity, and magnetic predictors. Notably, upward-continued Bouguer gravity, magnetic analytic signal, and the radiometric U/Th ratio ranked among the most influential variables in XGBoost but were comparatively less important in RF.

**Table 4:** Top 10 predictor features ranked by mean absolute SHAP values for the Random Forest and XGBoost models. SHAP values were calculated in probability space and reflect each predictor's average contribution to modelled VMS prospectivity.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

| **Rank** | **Random Forest Feature**             | **Mean \|SHAP\|** | **XGBoost Feature**               | **Mean \|SHAP\|** |
| -------- | ------------------------------------- | ----------------- | --------------------------------- | ----------------- |
| 1        | Radiometric Th/K alteration ratio     | 0.0501            | Radiometric Th/K alteration ratio | 0.0857            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 2        | Radiometric thorium (Th)              | 0.0397            | IDW-interpolated molybdenum (Mo)  | 0.0469            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 3        | IDW-interpolated molybdenum (Mo)      | 0.0339            | Raw zinc (Zn)                     | 0.0372            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 4        | Radiometric potassium (K)             | 0.0184            | Raw molybdenum (Mo)               | 0.0274            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 5        | IDW-interpolated zinc (Zn)            | 0.0184            | IDW-interpolated tin (Sn)         | 0.0262            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 6        | IDW-interpolated bismuth (Bi)         | 0.0147            | Upward-continued Bouguer gravity  | 0.0244            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 7        | IDW-interpolated lead (Pb)            | 0.0138            | Radiometric thorium (Th)          | 0.0228            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 8        | Raw zinc (Zn)                         | 0.0132            | Raw nickel (Ni)                   | 0.0224            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 9        | Gravity horizontal gradient magnitude | 0.0128            | Magnetic analytic signal          | 0.0219            |
| ---      | ---                                   | ---               | ---                               | ---               |
| 10       | Raw molybdenum (Mo)                   | 0.0126            | Radiometric U/Th ratio            | 0.0205            |
| ---      | ---                                   | ---               | ---                               | ---               |

These feature rankings identify three dominant predictor groups controlling modelled VMS prospectivity: (1) radiometric indicators of hydrothermal alteration, particularly the Th/K ratio; (2) geochemical pathfinder variables, notably Mo, Zn, Pb, Bi, Sn, and Ni; and (3) geophysical derivatives that capture lithological contrasts and structural architecture. The strong agreement between RF and XGBoost indicates that these predictor domains represent robust controls on VMS prospectivity within the Bathurst Mining Camp.

<<<<<<< HEAD
Sample-level SHAP attribution (beeswarm distributions; Fig. 9) clarifies the directional relationships between predictor values and modelled prospectivity across both classifiers. In the Random Forest model (Fig. 9a), elevated values of the radiometric Th/K alteration ratio consistently generate the strongest positive SHAP contributions, driving predictions toward mineralized classification. Conversely, lower airborne thorium concentrations correspond to positive SHAP values, capturing thorium mobility loss relative to potassium metasomatism in footwall alteration corridors enclosing feeder systems. High concentrations of molybdenum (both IDW-interpolated and raw), zinc, bismuth, and lead generate positive SHAP displacements, confirming that multi-element till dispersion trains directly increase target probability.

The XGBoost model (Fig. 9b) exhibits a complementary but broader dependency structure. Radiometric Th/K again provides the dominant positive displacement, reinforced by strong positive contributions from both interpolated and raw molybdenum, raw zinc, and raw nickel. Notably, potential-field structural signatures play a more prominent role in XGBoost: elevated upward-continued Bouguer gravity (500 m) and high magnetic analytic signal amplitudes both exert pronounced positive SHAP effects, reflecting the influence of dense volcanic stratigraphy and fault-bounded structural contacts on mineralization preservation. Across both algorithms, the concordance in feature directionality confirms that the machine learning models capture genuine geoscientific controls rather than algorithmic artifacts.

## **4.5 Model Robustness, Sensitivity, and Uncertainty Evaluation**

To assess the operational stability and geoscientific validity of the machine learning prospectivity framework, systematic sensitivity analyses were conducted evaluating class-label composition, geochemical feature sparsity, and spatial prediction uncertainty across the Bathurst Mining Camp.

### **4.5.1 Negative Label Ratio Sensitivity Analysis**

In data-driven prospectivity modeling, the composition of the negative class strongly influences decision boundaries and performance metrics. To evaluate model sensitivity to negative-label design, six distinct label configurations (A through F) were tested using 5-fold spatial block cross-validation (Table 6). For each configuration, Random Forest and XGBoost hyperparameters were independently optimized using 50 Optuna Bayesian optimization trials, and classification decision thresholds were calibrated at the validation-fold level.

**Table 6.** Negative label ratio sensitivity analysis under 5-fold spatial block cross-validation with 50-trial Optuna Bayesian hyperparameter optimization and fold-calibrated thresholds.

| Config | Negative Label Composition | n Barren | n Mahalanobis | RF ROC-AUC | RF Cal. BA | RF SR-AUC | RF Avg. Prec. | XGB ROC-AUC | XGB Cal. BA | XGB SR-AUC | XGB Avg. Prec. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | Pure random background (control) | 0 | 0 | 0.7918 ± 0.0596 | 0.5027 ± 0.0284 | 0.7520 ± 0.0765 | 0.4486 ± 0.1737 | 0.7901 ± 0.0711 | 0.7033 ± 0.1064 | 0.7514 ± 0.0859 | 0.4268 ± 0.1183 |
| **B** | Extreme feature-space dissimilarity | 0 | 250 | 0.9754 ± 0.0426 | 0.7394 ± 0.1594 | 0.8331 ± 0.1081 | 0.9736 ± 0.0502 | 0.9635 ± 0.0453 | 0.8135 ± 0.1652 | 0.8212 ± 0.0975 | 0.9500 ± 0.0612 |
| **C** | High Mahalanobis / Moderate barren | 63 | 187 | 0.8569 ± 0.0802 | 0.5358 ± 0.0904 | 0.7892 ± 0.0996 | 0.5452 ± 0.1604 | 0.8520 ± 0.0574 | 0.6949 ± 0.1082 | 0.7887 ± 0.0890 | 0.5879 ± 0.1018 |
| **D** | **Hybrid 1:1 ratio (Published)** | **125** | **125** | **0.7963 ± 0.0789** | **0.5515 ± 0.0703** | **0.7556 ± 0.0914** | **0.4605 ± 0.1407** | **0.7506 ± 0.1358** | **0.6791 ± 0.1464** | **0.7270 ± 0.1276** | **0.3828 ± 0.1301** |
| **E** | Moderate Mahalanobis / High barren | 187 | 63 | 0.7180 ± 0.0491 | 0.5198 ± 0.0423 | 0.6794 ± 0.0551 | 0.3755 ± 0.1784 | 0.7184 ± 0.0713 | 0.6092 ± 0.0662 | 0.6813 ± 0.0779 | 0.4760 ± 0.1257 |
| **F** | Confirmed barren drill holes only | 250 | 0 | 0.6742 ± 0.0350 | 0.5733 ± 0.0416 | 0.6264 ± 0.0151 | 0.4185 ± 0.1394 | 0.6314 ± 0.0555 | 0.6023 ± 0.1028 | 0.5936 ± 0.0356 | 0.4187 ± 0.1684 |

*Note:* Point-level cross-validation metrics evaluate discrete label separability. In the full camp continuous raster model (1,194,109 cells), the tuned hybrid model (Config D) achieves camp-scale RF ROC-AUC = 0.9318 ± 0.0368 and Success Rate AUC = 0.9680 (capturing 91.1% of known deposits in the top 10% study area).

As detailed in Table 6, when Mahalanobis pseudo-absences dominate the negative pool (Config B), point separability reaches near-perfection (RF ROC-AUC = 0.9754 ± 0.0426) because candidates are selected specifically for geometric separation in multi-parameter feature space. Conversely, relying exclusively on confirmed barren drill holes (Config F) yields lower apparent discrimination (RF ROC-AUC = 0.6742 ± 0.0350; XGBoost = 0.6314 ± 0.0555) because historical exploration drilling was concentrated within prospective volcanic horizons and structural corridors (hard negatives sharing favorable host-rock signatures). The balanced 125/125 hybrid configuration (Config D) establishes an operational compromise: it anchors the negative class to direct subsurface geological ground truth while incorporating statistical background representation, outperforming the pure random control (Config A: RF ROC-AUC = 0.7918 ± 0.0596) and preventing both the over-optimistic separability of Config B and the spatial bias of Config F.

### **4.5.2 Missingness Sensitivity Analysis for Sparse Pathfinder Elements**

Till geochemical datasets compiled across multi-campaign regional surveys frequently exhibit variable analytical coverage. Four critical pathfinder elements in the BMC compilation contain >50% null values at point sampling locations: Bismuth (Bi, 60.3% null), Indium (In, 60.3% null), Thallium (Tl, 60.3% null), and Manganese (Mn, 58.3% null). To determine whether preserving these sparse elements introduces unhelpful noise or delivers genuine predictive utility, we compared model performance with versus without these four raw point features, keeping the continuous IDW-interpolated surfaces in both models (Table 7).

**Table 7.** Missingness sensitivity analysis evaluating the impact of retaining versus excluding sparse raw point-geochemistry features (Bi, In, Tl, Mn) under 5-fold spatial block cross-validation for both Random Forest (RF) and XGBoost. IDW-interpolated surfaces are retained in both configurations. Results from `pipeline/03_training/sensitivity_missingness.py`.

| Configuration | Features | RF ROC-AUC (mean ± SD) | RF Avg Precision (mean ± SD) | RF Balanced Acc (mean ± SD) | XGB ROC-AUC (mean ± SD) | XGB Avg Precision (mean ± SD) | XGB Balanced Acc (mean ± SD) |
|---|---|---|---|---|---|---|---|
| With Bi/In/Tl/Mn raw columns | 60 | 0.7525 ± 0.0763 | 0.3826 ± 0.1712 | 0.4920 ± 0.0070 | 0.7473 ± 0.0732 | 0.3483 ± 0.1240 | 0.5378 ± 0.0677 |
| **Without Bi/In/Tl/Mn raw columns (Preferred)** | **56** | **0.7787 ± 0.0638** | **0.3890 ± 0.2158** | **0.4807 ± 0.0112** | **0.7734 ± 0.0527** | **0.4011 ± 0.1629** | **0.6102 ± 0.0855** |

Contrary to initial expectation, excluding the four sparse raw point features while retaining their IDW-interpolated surfaces improved model performance across both classifiers: RF ROC-AUC increased by +0.0262 (0.7525 → 0.7787) and XGBoost ROC-AUC increased by +0.0261 (0.7473 → 0.7734). XGBoost Average Precision improved by +0.0528. This consistent directional response indicates that raw point values for elements with >50–60% missingness introduce noise when median-imputed across geographically heterogeneous spatial blocks, and that the spatially continuous IDW surfaces derived from 2,753 unique regional till samples already capture the relevant geochemical contrast more robustly. Accordingly, the four sparse raw columns (bi_ppm, in_ppm, tl_ppm, mn_ppm) are excluded from the published 56-feature predictor matrix, while their IDW counterparts are retained.

### **4.5.3 Spatial Prediction Uncertainty and Alteration Footprint Coincidence**

To translate probabilistic prospectivity models into operational exploration decisions, model uncertainty was evaluated spatially by computing the fold-to-fold standard deviation ($\sigma$) of predicted prospectivity across the five spatial block cross-validation models (Fig. 7). High-prospectivity target corridors along the Tetagouche Group volcanic belts exhibit low uncertainty ($\sigma < 0.05$), indicating robust model consensus across geographically independent training partitions. In contrast, unprospective basement domains (Miramichi Group) show uniformly low prospectivity with near-zero standard deviation ($\sigma < 0.02$), confirming stable geological barrenness. Areas with elevated standard deviation ($\sigma > 0.15$) occur predominantly along structurally complex or covered margins where training data density is sparse, identifying priorities for infill geophysical and geochemical surveys.

The dominant predictive importance of the airborne radiometric Th/K ratio (mean |SHAP| = 0.0501 in RF, 0.0857 in XGBoost) was corroborated by regional spatial coincidence analysis (Fig. 8). Quantitative spatial intersection confirms that 42.2% (19 of 45) of documented VMS occurrences fall within the highest quartile of study-area Th/K values ($> 102.5$), representing a 1.7-fold enrichment over random spatial expectation ($p = 0.006$, one-sample proportion test), and 93.3% (42 of 45) occur above the regional median. This empirical coincidence validates that airborne gamma-ray spectrometry directly detects the potassium-depleted, thorium-retained hydrothermal alteration footprints enclosing BMC massive sulphide deposits.

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
# **5\. Discussion**

## **5.1 Controls on VMS Prospectivity in the Bathurst Mining Camp**

Feature importance analysis identified three dominant predictor domains: radiometric indicators of hydrothermal alteration, geochemical pathfinder signatures, and geophysical expressions of structural architecture. Among these, alteration-related radiometric variables emerged as the strongest controls on modelled VMS prospectivity.

The dominance of the Th/K alteration ratio is consistent with established models of VMS hydrothermal alteration, in which sericitic alteration commonly results in potassium enrichment relative to thorium within feeder zones and alteration halos (Franklin et al., 2005; Galley et al., 2007). Consequently, low Th/K ratios are commonly associated with hydrothermal alteration and can be used to trace potential fluid pathways (Shives et al., 1997). The consistently high ranking of both Th/K and thorium in the RF and XGBoost models suggests that alteration-related modification of the host rocks exerts a stronger control on prospectivity than individual geochemical pathfinders and was successfully captured by the machine-learning framework.

<<<<<<< HEAD
To reinforce the geological rationale for Th/K as the leading predictive feature (mean |SHAP| = 0.0501 in RF), we examined the spatial coincidence between known VMS deposits and regional Th/K anomalies. In the BMC, hydrothermal alteration associated with VMS formation typically involves intense quartz-sericite-chlorite alteration that depletes potassium while thorium remains relatively immobile in felsic volcanic footwall rocks (Shives et al., 1997; Lentz, 1999). Spatial extraction from the master survey raster reveals that 42.2% of known VMS occurrences (19 of 45) coincide with the top quartile of study-area Th/K values ($> 102.5$; Fig. 8), representing a 1.7-fold enrichment over random spatial expectation ($p = 0.006$, one-sample proportion test), and 93.3% (42 of 45) occur above the regional median. This confirms that airborne radiometric Th/K ratios directly detect the footwall alteration footprints enclosing BMC massive sulphide lenses.

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
Geochemical predictors provide complementary evidence for mineralization. Molybdenum was among the most influential variables in both classifiers, with both raw and IDW-interpolated Mo consistently ranked within the top predictors. This association is consistent with enrichment of Mo in high-temperature hydrothermal fluids and feeder-stockwork environments commonly associated with VMS systems (Franklin et al., 2005). Zinc-related variables also ranked highly, reflecting the central role of Zn within the polymetallic signature of BMC deposits. Additional pathfinder elements, including Pb, Bi, Sn, and Ni, suggest that the models capture a broad hydrothermal metal assemblage rather than isolated elemental anomalies.

The multivariate geochemical components derived from PCA and FA further reinforce this interpretation. PC1/FA1 represents the principal polymetallic VMS association, characterized by Zn-Pb-Cu-Ba enrichment together with associated Co and Ni, and exhibits the strongest relationship with known mineralization. This assemblage is consistent with the characteristic geochemical signatures reported for VMS deposits and the Bathurst Mining Camp (Franklin et al., 2005; Galley et al., 2007; Goodfellow & McCutcheon, 2003). Secondary components, including Bi-Cd enrichment (PC2/FA2), Ag-As variability (PC3), and Sn enrichment (PC4), define distinct geochemical populations that may reflect variations in hydrothermal processes, metal distribution, or mineralizing conditions across the camp. However, their geological significance appears to be secondary to the dominant polymetallic VMS signature represented by PC1/FA1.

Geophysical derivatives also contributed significantly to prospectivity prediction. Gravity horizontal gradients, upward-continued gravity responses, magnetic analytic signal amplitudes, and magnetic first vertical derivatives highlight the importance of lithological contacts and syn-volcanic fault systems that controlled fluid migration and sulfide deposition (Van Staal et al., 2003; Thomas et al., 2000). These results indicate that the most effective predictors represent an integrated alteration-mineralization-structure framework, consistent with established geological models for BMC VMS formation.

## **5.2 Implications for Camp-Scale Exploration Targeting**

Both machine-learning classifiers demonstrated strong predictive capability under spatially independent validation, confirming the effectiveness of integrating geophysical derivatives, multi-element till geochemistry, and geologically constrained training labels for regional-scale VMS targeting. Random Forest (RF) consistently outperformed XGBoost across ROC<sub>AUC</sub>, Average Precision, and Success Rate AUC metrics, indicating superior discrimination between mineralized and non-mineralized locations and greater efficiency in prioritizing exploration targets. The low fold-to-fold variability observed for both classifiers further indicates stable model generalization across geographically independent portions of the Bathurst Mining Camp (BMC), suggesting that the predictive relationships identified by the models are not restricted to individual deposit clusters.

<<<<<<< HEAD
The question of whether RF's advantage over XGBoost reflects a data distribution artefact or a fundamental algorithmic difference warrants discussion. Two mechanisms are likely at play. First, class imbalance handling: RF with `class_weight='balanced'` rescales the impurity criterion at each split proportionally to class frequency, penalising minority-class misclassifications globally throughout the ensemble. XGBoost's gradient loss adjusts positive-class gradients via boosting, but sequential error minimization in early iterations can establish majority-class-biased decision regions that subsequent trees must correct. With ~30–35 positive instances per training fold, this sequential correction may be less robust than RF's parallel bagging averaging. Second, spatial autocorrelation: RF's bootstrap aggregating decorrelates individual trees from local spatial clusters, producing probability estimates that generalize better across geographically holdout blocks. XGBoost, in contrast, may partially fit subtle spatial clusters in early boosting rounds, which is penalized more severely under spatial block CV where test blocks are geographically disjoint. Consequently, RF concentrates predictions tightly around confirmed structural-geochemical signatures, whereas XGBoost produces a broader prospectivity footprint that may be preferable for regional reconnaissance where false-negative risk must be minimized.

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
The RF and XGBoost prospectivity maps provide complementary spatial syntheses of the geological, geochemical, and geophysical information represented by the predictor dataset. The NNE-SSW alignment of high-prospectivity corridors identified by both RF and XGBoost reflects the strong influence of the regional volcanic and structural architecture of the Tetagouche Group on VMS distribution. The close correspondence between these corridors and known deposit clusters supports the geological validity of the prospectivity models and suggests that additional targets occurring along the same trend represent favourable exploration opportunities. Conversely, low-prospectivity regions correspond largely to geological units that are not considered favourable hosts for bimodal-siliciclastic VMS mineralization, indicating that the prospectivity patterns are consistent with the established geological framework of the BMC (Goodfellow, 2007; van Staal et al., 2003).

The RF model achieved particularly strong targeting performance, capturing 91.1% of known VMS deposits within the top 10% of the ranked study area, 97.8% within the top 20%, and 100% within the top 30%. These results demonstrate that the integrated workflow concentrates known mineralization into a relatively small proportion of the camp, thereby substantially reducing the search space for exploration. Reducing the search area while retaining a large proportion of known deposits is a principal objective of mineral prospectivity mapping and an important measure of map performance (Agterberg & Bonham-Carter, 2005; Carranza, 2008).

Several high-prospectivity anomalies identified by both classifiers occur outside the current inventory of known deposits while remaining spatially associated with favourable structural corridors, alteration signatures, and geochemical anomalies. Although the overall prospectivity patterns are broadly similar, the RF model delineates a more spatially focused set of target areas than XGBoost, consistent with its superior targeting efficiency under spatial cross-validation. These new anomalies represent compelling targets for follow-up exploration in the region or other covered terrain where conventional geological mapping may be less effective.

<<<<<<< HEAD
The strong predictive performance achieved by both classifiers also highlights the importance of representative negative training labels in mineral prospectivity mapping. A critical methodological consideration in data-driven prospectivity modeling is whether selecting negative samples based on feature-space dissimilarity could artificially inflate classification metrics or introduce circular reasoning by utilizing the predictor space both for label definition and classifier training. By definition, selecting candidate pseudo-absences with maximal Mahalanobis distance from the positive deposit centroid yields classes that are geometrically more separable in feature space by construction. As demonstrated in our sensitivity analysis (Table 6), training exclusively on Mahalanobis-dissimilar points yields an apparent ROC-AUC of 0.9754 ± 0.0426 (Config B), confirming that an unconstrained feature-dissimilar strategy leads to over-optimistic separability metrics.

However, our hybrid framework incorporates four essential methodological safeguards against circularity and overfitting:
1. **Covariance Anchor vs. Hyperplane Optimization:** The mathematical formulation of Mahalanobis distance ($D_M$; Eq. 6) evaluates candidate distance strictly against the positive-class centroid ($\boldsymbol{\mu}_+$) and positive-class covariance matrix ($\boldsymbol{\Sigma}_+$). It does not fit an optimal separating hyperplane, nor does it minimize cross-entropy loss or Gini impurity against candidate points, allowing the downstream tree-based algorithms (RF and XGBoost) to discover non-linear decision boundaries independently across the multi-source feature space.
2. **Ecological Habitat Suitability Analogy:** The method is functionally analogous to pseudo-absence selection in ecological niche and species distribution modeling (Barbet-Massin et al., 2012). In ecological modeling, background points drawn outside the environmental envelope of known occurrences represent unsuitable habitat rather than verified absences. In mineral systems modeling, drawing pseudo-absences outside the multi-element hydrothermal alteration and geophysical footprint ensures that background samples represent unmineralized regional crust rather than undetected ore deposits.
3. **Geological Ground-Truth Anchor:** Exactly half of the negative training pool ($n = 125$) consists of New Brunswick Geological Survey diamond drill holes confirmed by subsurface core logging to be completely barren of economic mineralization. This provides an empirical geological anchor that is entirely independent of the geophysical and geochemical feature space.
4. **Empirical Bounds via Controlled Experiments:** As documented in Table 6, comparing the hybrid configuration (Config D) against a pure random background control (Config A) demonstrates that the hybrid approach provides a robust, geologically coherent improvement over random background sampling without the extreme, artificial separability observed when training exclusively on feature-dissimilar points (Config B: AUC = 0.9754).
=======
The strong predictive performance achieved by both classifiers also highlights the importance of representative negative training labels in mineral prospectivity mapping. The hybrid class label strategy adopted in this study, which combines confirmed barren drill intercepts with feature-space dissimilar background samples, provided a more geologically meaningful representation of negative conditions than conventional random background sampling. This interpretation is consistent with the recent work by Parsa and Cumani (2025), who demonstrated that the representativeness of negative class labels can significantly influence classification performance and the spatial selectivity of prospectivity models. The low variability observed across spatial cross-validation folds and the high targeting efficiency of the RF model suggest that geologically constrained negative-label selection contributed to robust model performance and improved camp-scale exploration targeting.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

## **5.3 Limitations and Future Directions**

Despite the strong model performance, several limitations are acknowledged. The modelling framework was trained using a relatively small set of known VMS deposits, reflecting the finite inventory of documented occurrences within the BMC. Although the hybrid class-label strategy improved class separation and validation stability, uncertainty inevitably remains in poorly explored areas where the true distribution of mineralization is unknown.

<<<<<<< HEAD
A further limitation is that no formal spatial declustering or thinning was applied to the raw multi-campaign till point database prior to merging. While IDW interpolation to a continuous 50 m raster surface, nearest-neighbour joins with 1,000 m limits, and large spatial block CV partitions (spanning 20–30 km each) substantially mitigate point-clustering artifacts, future studies should evaluate the effect of cell-declustering or distance-based spatial thinning on multi-campaign geochemical compilations.

In addition, characterizing prediction uncertainty is critical for translating machine-learning prospectivity maps into operational exploration decisions. To quantify uncertainty, we computed the fold-to-fold standard deviation ($\sigma$) of predicted prospectivity across the five spatial block cross-validation models (Fig. 7). High-PI target zones along the Tetagouche Group structural corridors exhibit low $\sigma$ ($< 0.05$), demonstrating that these targets are robustly predicted regardless of which geographic block was withheld during training. In contrast, low-PI regions coinciding with the Miramichi Group basement and Four Falls Group show consistently low PI with minimal $\sigma$, confirming genuine geological barrenness. Low-PI zones within the Tetagouche belt displaying elevated $\sigma$ should be interpreted as under-sampled rather than definitively barren, representing priority candidates for infill geophysical surveys.
=======
The prospectivity framework is also constrained by the datasets available at camp scale. Future work could also explore the integration of multiple classifier outputs through ensemble prospectivity modelling and uncertainty quantification to better characterize areas of agreement and disagreement between machine-learning predictions.
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

Notwithstanding these limitations, the results demonstrate that integrating radiometric alteration signatures, structural geophysical derivatives, and multi-element till geochemistry within a machine-learning framework provides an effective approach for camp-scale VMS exploration targeting in covered terranes.

# **6\. Conclusions**

<<<<<<< HEAD
This study demonstrates the effectiveness of integrating geophysical derivatives, multi-element till geochemistry, and machine learning for camp-scale volcanogenic massive sulphide (VMS) prospectivity mapping in the Bathurst Mining Camp (BMC), New Brunswick. Spatially independent validation showed that both Random Forest (RF) and XGBoost successfully captured the principal geological controls on VMS mineralization, confirming the value of combining alteration, geochemical, and structural datasets within a unified prospectivity framework.

Among the evaluated classifiers, RF provided the strongest overall exploration-targeting performance, achieving a ROC<sub>AUC</sub> of 0.9318 ± 0.0368 and a success-rate AUC of 0.9680. Furthermore, 91.1% of known VMS deposits were captured within the highest-ranked 10% of the study area, indicating that the model can substantially reduce the exploration search space while maintaining high deposit recovery.

Feature importance analysis revealed strong agreement between RF and XGBoost regarding the principal controls on prospectivity. Radiometric indicators of hydrothermal alteration, particularly the Th/K ratio, emerged as the most influential predictors, followed by molybdenum- and zinc-related geochemical variables and gravity- and magnetic-derived structural attributes. Together, these findings highlight the fundamental roles of hydrothermal alteration, metal dispersion, and structural architecture in controlling the distribution of VMS mineralization within the BMC.

Prospectivity maps generated by both RF and XGBoost delineate several high-priority targets beyond the current inventory of known deposits while remaining consistent with established geological, geochemical, and structural controls. Despite broadly similar spatial patterns, RF produced a more focused distribution of high-prospectivity zones and superior exploration-targeting performance, supporting its recommendation as the preferred model for camp-scale VMS exploration in the BMC.

Importantly, we have demonstrated the value of a geologically constrained hybrid class-label framework that combines confirmed barren drill intercepts with feature-space dissimilar background samples to represent non-mineralized conditions. The overall results show that integrating geophysical derivatives, multi-element till geochemistry, and geologically informed class labels within a spatially validated machine-learning workflow provides an effective, transferable, and optimization-aware approach for camp-scale VMS exploration targeting in mature and partially covered mining districts, contributing to the broader development of next-generation Exploration Information Systems (Daviran et al., 2026).

# **Figure Captions**
=======
# This study demonstrates the effectiveness of integrating geophysical derivatives, multi-element till geochemistry, and machine learning for camp-scale volcanogenic massive sulphide (VMS) prospectivity mapping in the Bathurst Mining Camp (BMC), New Brunswick. Spatially independent validation showed that both Random Forest (RF) and XGBoost successfully captured the principal geological controls on VMS mineralization, confirming the value of combining alteration, geochemical, and structural datasets within a unified prospectivity framework

# Among the evaluated classifiers, RF provided the strongest overall exploration-targeting performance, achieving a ROC<sub>AUC</sub> of 0.9318 ± 0.0368 and a success-rate AUC of 0.968. Furthermore, 91.1% of known VMS deposits were captured within the highest-ranked 10% of the study area, indicating that the model can substantially reduce the exploration search space while maintaining high deposit recovery

# Feature importance analysis revealed strong agreement between RF and XGBoost regarding the principal controls on prospectivity. Radiometric indicators of hydrothermal alteration, particularly the Th/K ratio, emerged as the most influential predictors, followed by molybdenum- and zinc-related geochemical variables and gravity- and magnetic-derived structural attributes. Together, these findings highlight the fundamental roles of hydrothermal alteration, metal dispersion, and structural architecture in controlling the distribution of VMS mineralization within the BMC

# Prospectivity maps generated by both RF and XGBoost delineate several high-priority targets beyond the current inventory of known deposits while remaining consistent with established geological, geochemical, and structural controls. Despite broadly similar spatial patterns, RF produced a more focused distribution of high-prospectivity zones and superior exploration-targeting performance, supporting its recommendation as the preferred model for camp-scale VMS exploration in the BMC

# Importantly, we have demonstrated the value of a geologically constrained hybrid class-label framework that combines confirmed barren drill intercepts with feature-space dissimilar background samples to represent non-mineralized conditions. The overall results show that integrating geophysical derivatives, multi-element till geochemistry, and geologically informed class labels within a spatially validated machine-learning workflow provides an effective and transferable approach for camp-scale VMS exploration targeting in mature and partially covered mining districts

**Figure Captions**
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05

**Fig. 1** The geology of the BMC with major lithostratigraphic units, structural trends and locations of known VMS deposits (e.g. Brunswick No. 12 (B12) and Brunswick No. 6 (B6)) deposits. Inset: Approximate location of the BMC, northern New Brunswick, Canada. Modified from van Staal et al. (2003).

**Fig. 2** Schematic workflow chart of the machine learning prospectivity mapping pipeline. Boxes denote processing stages; arrows denote data flow. CLR centered log-ratio; CoDA compositional data analysis; FA factor analysis; IDW inverse distance weighting; MEAS multi-element anomaly score; PCA principal component analysis; SMOTE synthetic minority over-sampling technique; RF random forest; XGBoost extreme gradient boosting; and SHAP SHapley Additive exPlanations.

**Fig. 3** Geophysical and geochemical input datasets for the BMC obtained from NBDNR through ArcGIS REST services; **(a–e)** Airborne residual total magnetic intensity (RTMI), Bouguer gravity, uranium (eU ppm), thorium (eTh ppm) and potassium (%K); **(f-j)** Selected till-geochemistry point data for lead (Pb ppm), zinc (Zn ppm), copper (Cu ppm), silver (Ag) and antimony (Sb).

**Fig. 4.** Spatial cross-validation fold assignment used for model evaluation. Labelled locations were partitioned into five geographically independent folds based on easting-coordinate quantiles, yielding approximately equal sample counts per fold. In each cross-validation iteration, one fold was reserved for validation while the remaining four folds were used for model training. The residual magnetic intensity layer shown in the background provides geological context and illustrates the spatial distribution of validation folds across the Bathurst Mining Camp.

**Fig. 5.** Success-rate curves for the Random Forest (RF) and XGBoost classifiers under spatial cross-validation. Curves show the cumulative percentage of known VMS deposits captured as a function of the cumulative percentage of the study area ranked by prospectivity. The RF model demonstrates superior targeting efficiency, capturing 91.1% of known deposits within the highest-ranked 10% of the study area and achieving a larger Success Rate AUC than XGBoost.

**Fig. 6.** Camp-scale VMS prospectivity maps generated using (a) Random Forest and (b) XGBoost models. Prospectivity Index (PI) values represent the predicted probability of VMS mineralization. Dotted circles indicate known VMS deposits, and outlined polygons represent high-priority target areas (PI > 0.7). Both models delineate NNE-SSW trending prospective corridors. The Random Forest model produces a more spatially focused distribution of high-prospectivity zones.

<<<<<<< HEAD
**Fig. 7.** Camp-scale spatial uncertainty map displaying the fold-to-fold standard deviation ($\sigma$) of predicted Random Forest prospectivity index values across the five spatial block cross-validation models. High-prospectivity target corridors along the Tetagouche Group display low uncertainty ($\sigma < 0.05$), indicating high-confidence drill-targeting potential. Low-prospectivity basement units exhibit consistently low $\sigma$ ($< 0.02$), confirming robust barrenness. Zones of elevated $\sigma$ ($> 0.15$) delineate structurally complex or covered areas with sparse label support, identifying priority targets for infill geophysical and geochemical data acquisition.

**Fig. 8.** Regional airborne radiometric Thorium/Potassium (Th/K) alteration ratio grid across the Bathurst Mining Camp with documented VMS occurrences overlaid as circular markers. High Th/K values (warm tones) delineate regional potassium-depleted, thorium-retained hydrothermal alteration corridors along the Cambro-Ordovician Tetagouche Group. Quantitative spatial intersection confirms that 42.2% (19 of 45) of known VMS deposits fall within the top quartile of study-area Th/K values ($> 102.5$), representing a 1.7-fold spatial enrichment over random uniform expectation ($p = 0.006$), and 93.3% (42 of 45) of known deposits occur above the study-area median.

**Fig. 9.** SHAP (SHapley Additive exPlanations) beeswarm feature attribution summaries for the top 15 predictors in the (a) Random Forest and (b) XGBoost models. Each point represents an individual training sample; horizontal position indicates the SHAP attribution value (contribution to predicted VMS probability, where positive values displace predictions toward mineralized classification), and point colour denotes relative feature value (red = high, blue = low). Both classifiers demonstrate consistent geological directionality: elevated radiometric Th/K alteration ratios, depleted thorium, enriched pathfinder concentrations (Mo, Zn, Bi, Pb, Sn, Ni), and elevated potential-field structural gradients consistently drive positive prospectivity predictions.

# **Appendix A. Dataset Feature Inventory**

**Table A1.** Complete geoscientific feature inventory (102 features) integrated into the machine learning prospectivity database for the Bathurst Mining Camp.

| Category | Count | Features / Variable Description | Processing & Transformation |
|---|---|---|---|
| **Raw Till Geochemistry** | 17 | Ag, As, Ba, Bi, Cd, Co, Cu, Fe, In, Mn, Mo, Ni, Pb, Sb, Sn, Tl, Zn (ppm) | Nearest-neighbour join within 1,000 m radius; median imputed per CV fold |
| **Log-Transformed Raw Geochem** | 17 | $\ln(\text{element} + 1)$ for all 17 raw geochemical elements | Variance stabilization for right-skewed geochemical distributions |
| **IDW Continuous Surfaces** | 17 | Spatially interpolated surfaces for all 17 till elements | Inverse Distance Weighting ($p = 2$, $n = 12$ nearest neighbours, 100 m cell) |
| **Log-Transformed IDW Geochem** | 17 | $\ln(\text{IDW\_element} + 1)$ for all 17 interpolated surfaces | Continuous log-scale geochemical intensity fields |
| **Multivariate Geochemical Comps** | 8 | Compositional PC1–PC4 and FA1–FA4 | Centered log-ratio (CLR) transform followed by PCA and Varimax Factor Analysis |
| **Geochemical Composite Score** | 1 | Geologically Weighted Multi-Element Anomaly Score (MEAS) | Standardized z-score composite weighted by BMC metallogenic literature |
| **Airborne Radiometrics** | 6 | Total Count, eU (ppm), eTh (ppm), %K, and radioelement ratios (K/Th, U/Th, Th/K) | 256-channel gamma-ray spectrometry compilation |
| **Airborne Magnetics & Derivatives** | 8 | RTMI, First Vertical Derivative (FVD), Total Horizontal Gradient (THG), Tilt Derivative (TDR), Analytic Signal (AS), Upward Continuations (500 m, 1,000 m) | Fourier-domain native-grid derivative calculations prior to 100 m rasterization |
| **Ground/Airborne Gravity & Derivs** | 5 | Bouguer Gravity Anomaly, Gravity THG, Gravity AS, Upward Continuations | Regional gravity compilation with native gradient filtering |
| **Spatial & Coordinate Attributes** | 6 | Easting, Northing, Geographic Quadrant, Fold ID, Guard Distance, Deposit ID | Stratified partitioning and spatial tracking (excluded from classifier feature matrix) |
| **Total Features** | **102** | **Predictor matrix: 60 active numeric ML features** | |

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
# **References**

Agterberg, F.P., Bonham-Carter, G.F. (2005). Measuring the Performance of Mineral-Potential Maps. _Natural Resources Research_ 14, 1–17. <https://doi.org/10.1007/s11053-005-4674-0>

Aitchison, J. (1986). _The statistical analysis of compositional data_. Chapman and Hall.

<<<<<<< HEAD
Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In _Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining_ (pp. 2623–2631). ACM. <https://doi.org/10.1145/3292500.3330701>

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
Barbet-Massin, M., Jiguet, F., Albert, C. H., & Thuiller, W. (2012). Selecting pseudo-absences for species distribution models: how, where and how many? _Methods in Ecology and Evolution, 3_(2), 327–338. <https://doi.org/10.1111/j.2041-210X.2011.00172.x>

Blakely, R. J. (1995). _Potential theory in gravity and magnetic applications_. Cambridge University Press.

Bonham-Carter, G. F. (1994). _Geographic information systems for geoscientists: Modelling with GIS_. Pergamon Press.

Breiman, L. (2001). Random forests. _Machine Learning, 45_(1), 5–32. <https://doi.org/10.1023/A:1010933404324>

Brenning, A. (2012). Spatial cross-validation and bootstrap for the assessment of prediction rules in remote sensing: The R package sperrorest. _2012 IEEE International Geoscience and Remote Sensing Symposium_, 5372–5375. <https://doi.org/10.1109/IGARSS.2012.6352393>

Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. In _Proceedings of the 20th International Conference on Pattern Recognition (ICPR)_ (pp. 3121–3124). IEEE. <https://doi.org/10.1109/ICPR.2010.764>

Cardoso-Fernandes, J., Lima, J., Lima, A., Roda-Robles, E., Köhler, M., Schaefer, S., Barth, A., Knobloch, A., Gonçalves, M. A., Gonçalves, F., & Teodoro, A. C. (2022). Stream sediment analysis for Lithium (Li) exploration in the Douro region (Portugal): A comparative study of the spatial interpolation and catchment basin approaches. _Journal of Geochemical Exploration_, _236_, 106978. <https://doi.org/10.1016/j.gexplo.2022.106978>

Carranza, E.J.M. (2008). Geochemical Anomaly and Mineral Prospectivity Mapping in GIS. Handbook of Exploration and Environmental Geochemistry, Vol. 11. Elsevier, Amsterdam.

Carranza, E. J. M., & Laborte, A. G. (2015). Random forest predictive modeling of mineral prospectivity with small number of prospects and data with missing values in Abra (Philippines). _Computers & Geosciences, 74_, 60–70. <https://doi.org/10.1016/j.cageo.2014.10.004>

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. _Journal of Artificial Intelligence Research, 16_, 321–357. <https://doi.org/10.1613/jair.953>

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, 785–794. <https://doi.org/10.1145/2939672.2939785>

<<<<<<< HEAD
Daviran, M., Maghsoudi, A., Ghezelbash, R., & Pradhan, B. (2021). A new strategy for spatial predictive mapping of mineral prospectivity: Automated hyperparameter tuning of random forest approach. _Computers & Geosciences, 148_, 104688. <https://doi.org/10.1016/j.cageo.2021.104688>

Daviran, M., Maghsoudi, A., & Ghezelbash, R. (2025). Optimized AI-MPM: Application of PSO for tuning the hyperparameters of SVM and RF algorithms. _Computers & Geosciences, 195_, 105785. <https://doi.org/10.1016/j.cageo.2024.105785>

Daviran, M., & Maghsoudi, A. (2026). Optimized unsupervised AI-MPM: Application of genetic algorithm for optimization of Fuzzy c-means clustering performance for targeting porphyry copper deposits. _Physics and Chemistry of the Earth_, 104463. <https://doi.org/10.1016/j.pce.2026.104463>

Daviran, M., Maghsoudi, A., & Yousefi, M. (2026). Analyzing the variety of optimization algorithms and its effect on unsupervised mineral prospectivity modeling; A proposal for the future improvement of exploration information system (EIS). _Ore Geology Reviews_, 107291. <https://doi.org/10.1016/j.oregeorev.2026.107291>

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In _Proceedings of the 23rd International Conference on Machine Learning (ICML '06)_ (pp. 233–240). ACM. <https://doi.org/10.1145/1143844.1143874>

Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barceló-Vidal, C. (2003). Isometric logratio transformations for compositional data analysis. _Mathematical Geology, 35_(3), 279–300. <https://doi.org/10.1023/A:1023818214614>

Fawcett, T. (2006). An introduction to ROC analysis. _Pattern Recognition Letters, 27_(8), 861–874. <https://doi.org/10.1016/j.patrec.2005.10.010>

Filzmoser, P., Hron, K., & Reimann, C. (2009). Principal component analysis for compositional data with outliers. _Environmetrics, 20_(6), 621–632. <https://doi.org/10.1002/env.966>

Filzmoser, P., Hron, K., & Templ, M. (2018). _Applied Compositional Data Analysis: With R examples_. Springer. <https://doi.org/10.1007/978-3-319-96422-5>

Franklin, J. M., Gibson, H. L., Jonasson, I. R., & Galley, A. G. (2005). Volcanogenic massive sulphide deposits. _Economic Geology, 100th Anniversary Volume_, 523–560. <https://doi.org/10.5382/AV100.17>

Galley, A. G., Hannington, M. D., & Jonasson, I. R. (2007). Volcanogenic massive sulphide deposits. In W. D. Goodfellow (Ed.), _Mineral deposits of Canada: A synthesis of major deposit-types, district metallogeny, the evolution of geological provinces, and the exploration methods_ (Special Publication No. 5, pp. 141–161). Geological Association of Canada, Mineral Deposits Division.

Ghane, B., Lentz, D.R., Thorne, K.G., Ugalde, H. A., (2026_)._ Calibration of Airborne Geophysical Data with In Situ Petrophysical Measurements for Mineral Prospectivity Mapping Using XGBoost Random Forest. _Nat Resour Res_ **35**, 245–277 (2026). <https://doi.org/10.1007/s11053-025-10579-7>

Goodfellow, W. D. (2007). Metallogeny of the Bathurst Mining Camp, northern New Brunswick. In W. D. Goodfellow (Ed.), _Mineral deposits of Canada: A synthesis of major deposit-types, district metallogeny, the evolution of geological provinces, and the exploration methods_ (Special Publication No. 5, pp. 443–469). Geological Association of Canada, Mineral Deposits Division.

Goodfellow, W. D., & McCutcheon, S. R. (2003). Geologic and genetic attributes of volcanic-associated massive sulphide deposits of the Bathurst Mining Camp, northern New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), _Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine_ (Economic Geology Monograph No. 11, pp. 19–60). Society of Economic Geologists. <https://doi.org/10.5382/Mono.11.13>

<<<<<<< HEAD
Lentz, D. R. (1999). Petrology, geochemistry, and oxygen isotope interpretation of felsic volcanic and related rocks hosting the Brunswick-type VMS deposits (Brunswick no. 12, Brunswick no. 6, and Austin Brook), Bathurst Mining Camp, New Brunswick. _Economic Geology_, 94(1), 57–86. <https://doi.org/10.2113/gsecongeo.94.1.57>

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
Li, T., Xia, Q., Zhao, M., Gui, Z., & Leng, S. (2020). Prospectivity mapping for tungsten polymetallic mineral resources, Nanling Metallogenic Belt, South China: Use of Random Forest algorithm from a perspective of data imbalance. _Natural Resources Research_, _29_(1), 203–227. <https://doi.org/10.1007/s11053-019-09564-8>

Lundberg, S. M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. _Advances in Neural Information Processing Systems, 30_, 4765–4774.

Luo, Z., Farahbakhsh, E., Hore, S., and Müller, R. D. (2026). DEEP-SEAM: an explainable semi-supervised deep learning framework for mineral prospectivity mapping, Geosci. Model Dev., 19, 2593–2625, <https://doi.org/10.5194/gmd-19-2593-2026>.

Maepa, F., Smith, R. S., & Tessema, A. (2021). Support vector machine and artificial neural network modelling of orogenic gold prospectivity mapping in the Swayze greenstone belt, Ontario, Canada. _Ore Geology Reviews, 139_, 104408. <https://doi.org/10.1016/j.oregeorev.2020.103968>

Mahalanobis, P.C. (1936). On the generalized distance in statistics. _Proceedings of the National Institute of Sciences of India_, 2, 49–55.

Mami Khalifani, F., Lentz, D.R. & Walker, J.A. (2025) Machine learning-based mineral prospectivity mapping of epithermal gold mineralization in Northern New brunswick: a comparative study of random forest, support vector machine, and XGboost classifiers. _Earth Sci Inform_ **18**, 553.<https://doi.org/10.1007/s12145-025-02041-2>

McClenaghan, M. B., Paulen, R. C., Smith, I. R., Rice, J. M., Plouffe, A., McMartin, I., Campbell, J. E., Lehtonen, M., Parsasadr, M., & Beckett-Brown, C. E. (2023). Review of till geochemistry and indicator mineral methods for mineral exploration in glaciated terrain. _Geochemistry: Exploration, Environment, Analysis_, _23_(4), geochem2023-013. <https://doi.org/10.1144/geochem2023-013>

<<<<<<< HEAD
McCutcheon, S. R., Luff, W. M., & Lentz, D. R. (2003). Context of zinc-lead massive sulfide deposits in the Bathurst Mining Camp, New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), _Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine_ (Economic Geology Monograph No. 11, pp. 1–18). Society of Economic Geologists. <https://doi.org/10.5382/Mono.11.01>

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
McCutcheon, S. R., & Walker, J. A. (2020). Great mining camps of Canada 8. The Bathurst Mining Camp, New Brunswick, Part 2: Mining history and contributions to society. _Geoscience Canada, 47_(3), 143–166. <https://doi.org/10.12789/geocanj.2020.47.163>.

Miller, H. G., & Singh, V. (1994). Potential field tilt—A new concept for location of potential field sources. _Journal of Applied Geophysics, 32_(2–3), 213–217. [https://doi.org/10.1016/0926-9851(94)90022-1](https://doi.org/10.1016/0926-9851%2894%2990022-1)

Natural Resources Canada. (2024). _Canada's critical minerals list 2024_. Government of Canada. [https://www.canada.ca/en/natural-resources-canada/news/2024/06/government-of-canada-releases-updated-critical-minerals-list.html. Accessed 2026](https://www.canada.ca/en/natural-resources-canada/news/2024/06/government-of-canada-releases-updated-critical-minerals-list.html.%20Accessed%202026).

Nidhi, D.K., Mohapatra, S.K., Nevalainen, P., [Heikkonen](https://link.springer.com/article/10.1007/s10791-026-10192-z#auth-Jukka-Heikkonen-Aff1), [J](https://link.springer.com/article/10.1007/s10791-026-10192-z#auth-Jukka-Heikkonen-Aff1)., & Kanth, R.,(2026_)._ Mineral prospectivity mapping under extreme imbalance using contrastive embeddings, balanced learning and integrated uncertainty analysis. _Discov Computing_ **29**, 288. <https://doi.org/10.1007/s10791-026-10192-z>

Nykänen, V., Groves, D. I., Ojala, V. J., Eilu, P., & Gardoll, S. J. (2008). Reconnaissance-scale conceptual fuzzy-logic prospectivity modelling for iron oxide copper–gold deposits in the northern Fennoscandian Shield, Finland. _Australian Journal of Earth Sciences, 55_(1), 25–38. <https://doi.org/10.1080/08120090701581372>

Nykänen, V., Lahti, I., Niiranen, T., & Korhonen, K. (2015). Receiver operating characteristics (ROC) as validation tool for prospectivity models—A magmatic Ni–Cu case study from the Central Lapland Greenstone Belt, Northern Finland. Ore Geology Reviews, 71, 853–860. <https://doi.org/10.1016/j.oregeorev.2014.09.007>

Parkhill, M. A., & Doiron, A. (2003). Quaternary geology and till geochemistry of the Bathurst Mining Camp, New Brunswick. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), _Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine_ (Economic Geology Monograph No. 11, pp. 101–122). Society of Economic Geologists <https://doi.org/10.5382/Mono.11.28>

Parsa, M. (2021). A data augmentation approach to XGboostbased mineral potential mapping: An example of carbonate hosted Zn-Pb mineral systems of Western Iran. Journal of Geochemical Exploration, 228, 106811. <https://doi.org/10.1016/j.gexplo.2021.106811>

Parsa, M., & Carranza, E. J. M. (2021). Modulating the impacts of stochastic uncertainties linked to deposit locations in data-driven predictive mapping of mineral prospectivity. Natural Resources Research, 30(5), 3081–3097. <https://doi.org/10.1007/s11053-021-09891-9>

Parsa, M., Lentz, D. R., & Walker, J. A. (2023). Predictive modeling of prospectivity for VHMS mineral deposits, northeastern Bathurst Mining Camp, NB, Canada, using an ensemble regularization technique. _Natural Resources Research, 32_, 19–36. <https://doi.org/10.1007/s11053-022-10133-9>

Parsa, M., Cumani, R. (2025). Class Label Representativeness in Machine Learning-Based Mineral Prospectivity Mapping. _Natural Resources Research_ 34, 1901–1925. <https://doi.org/10.1007/s11053-025-10468-z>

Pham, L. T., Eldosouky, A. M., Oksum, E., & Saada, S. A. (2022). A new high resolution filter for source edge detection of potential field data. _Geocarto International_, 37(11), 3051–3068. <https://doi.org/10.1080/10106049.2020.1849414>

<<<<<<< HEAD
QGIS Development Team. (2026). _QGIS Geographic Information System (Version 4.2.0 "Belém do Pará")_. Open Source Geospatial Foundation Project. <https://qgis.org>

=======
>>>>>>> 409badf4b7e147508c94e920d166099ff9266e05
Reimann, C., Filzmoser, P., Garrett, R. G., & Dutter, R. (2008). _Statistical data analysis explained: Applied environmental statistics with R_. Wiley. <https://doi.org/10.1002/9780470987605>

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, or phylogenetic structure. _Ecography, 40_(8), 913–929. <https://doi.org/10.1111/ecog.02881>

Rodriguez-Galiano, V. F., Chica-Olmo, M., & Chica-Rivas, M. (2014). Predictive modelling of gold potential with the integration of multisource information based on random forest: a case study on the Rodalquilar area, Southern Spain. International Journal of Geographical Information Science, 28(7), 1336–1354. <https://doi.org/10.1080/13658816.2014.885527>

Roest, W. R., Verhoef, J., & Pilkington, M. (1992). Magnetic interpretation using the 3-D analytic signal. _Geophysics, 57_(1), 116–125. <https://doi.org/10.1190/1.1443174>

Rogers, N., & van Staal, C. R., (2003). Volcanology and Tectonic Setting of the Northern Bathurst Mining Camp: Part II. Mafic Volcanic Constraints on Back-Arc Opening. In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), _Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine_ (Economic Geology Monograph No. 11, pp. 61–100). Society of Economic Geologists. <https://doi.org/10.5382/Mono.11.10>

Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. In _Proceedings of the 1968 23rd ACM National Conference_ (pp. 517–524). ACM. <https://doi.org/10.1145/800186.810616>

Shives, R.B.K., Charbonneau, B.W., and Ford, K.L., 1997, The detection of potassic alteration by gamma-ray spec-trometry–recognition of alteration related to mineralization, _in_ A.G. Gubins, _ed.,_ Proceedings of Exploration 97, Fourth Decennial International Conference on Mineral Exploration: Geophysics and Geochemistry at the Millennium, p. 741-752 .

Sun, T., Chen, F., Zhong, L., Liu, W., Wang, Y. (2019). GIS-based mineral prospectivity mapping using machine learning methods: A case study from Tongling ore district, eastern China. Ore Geology Reviews 109 (2019) 26–49. <https://doi.org/10.1016/j.oregeorev.2019.04.003>

Thomas, M. D., Walker, J. A., Keating, P. B., Shives, R. B. K., Kiss, F. G. & Goodfellow, W. D. (2000). Geophysical atlas of massive sulphide signatures Bathurst mining camp, New Brunswick. _Geological Survey of Canada, Open File_, 3887, 105. <https://doi.org/10.4095/211549>

Ugalde, H., Morris, W. A., & van Staal, C. R. (2019). The Bathurst Mining Camp, New Brunswick: data integration, geophysical modelling, and implications for exploration. _Canadian Journal of Earth Sciences, 56_(5), 433–451. <https://doi.org/10.1139/cjes-2018-0048>

van Staal, C. R., Wilson, R. A., Rogers, N., Fyffe, L. R., Langton, J. P., McCutcheon, S. R., McNicoll, V., & Ravenhurst, C. E. (2003). [Geology and Tectonic History of the Bathurst Supergroup, Bathurst Mining Camp, and Its Relationships to Coeval Rocks in Southwestern New Brunswick and Adjacent Maine—A Synthesis](https://pubs.geoscienceworld.org/books/edited-volume/1217/chapter/107019322/Geology-and-Tectonic-History-of-the-Bathurst). In W. D. Goodfellow, S. R. McCutcheon, & J. M. Peter (Eds.), _Massive sulphide deposits of the Bathurst Mining Camp, New Brunswick, and northern Maine_ (Economic Geology Monograph No. 11, pp. 37–60). Society of Economic Geologists. <https://doi.org/10.5382/Mono.11.03>

Verduzco, B., Fairhead, J. D., Green, C. M., & MacKenzie, C. (2004). New insights into magnetic derivatives for structural mapping. _The Leading Edge, 23_(2), 116–119. <https://doi.org/10.1190/1.1651454>

Zuo, R., Kreuzer, O.P., Wang, J., Xiong, Y., Zhang, Z. and Wang, Z., 2021. Uncertainties in GIS-based mineral prospectivity mapping: Key types, potential impacts and possible solutions. Natural Resources Research, 30(5), pp.3059-3079. <https://link.springer.com/article/10.1007/s11053-021-09871-z>

Zuo, R., Xiong, Y., Wang, Z., & Carranza, E. J. M. (2019). Deep learning and its application in geochemical mapping. _Earth-Science Reviews, 192_, 1–14. <https://doi.org/10.1016/j.earscirev.2019.02.023>
