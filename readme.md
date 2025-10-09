\# 🌌 PhyDAE: Physics-Guided Degradation-Adaptive Experts for All-in-One Remote Sensing Image Restoration



<p align="center">

&nbsp; <img src="assets/teaser.png" width="90%"/>

</p>



<p align="center">

&nbsp; <b>Official PyTorch implementation of</b><br>

&nbsp; <i>"PhyDAE: Physics-Guided Degradation-Adaptive Experts for All-in-One Remote Sensing Image Restoration"</i><br>

&nbsp; <b>Zhe Dong</b>, Yuzhe Sun, Haochen Jiang, Tianzhu Liu, <b>Yanfeng Gu\*</b><br>

&nbsp; Harbin Institute of Technology · Heilongjiang Province Key Laboratory of Space–Air–Ground Integrated Intelligent Remote Sensing

</p>



---



\## 🛰️ Overview



\*\*PhyDAE\*\* introduces a \*\*physics-guided, degradation-adaptive expert framework\*\* for unified remote sensing image restoration.  

Unlike prior “black-box” all-in-one models, PhyDAE bridges \*\*physical imaging principles\*\* with \*\*learnable expert mechanisms\*\*, achieving \*\*interpretable, efficient, and physically consistent restoration\*\*.



<p align="center">

&nbsp; <img src="assets/flowchart.pdf" width="95%">

</p>



> The two-stage cascaded architecture transforms degradation cues from \*implicit features\* into \*explicit decision signals\*, enabling precise differentiation and handling of heterogeneous degradations such as \*\*haze, noise, blur, and low-light\*\*.



---



\## ⚙️ Key Features



✅ \*\*Physics-Guided Degradation Modeling\*\*  

Incorporates atmospheric, sensor, and illumination physics via frequency-domain constraints.



✅ \*\*Two-Stage Progressive Restoration\*\*  

Residual manifold projection (RMP) + frequency-aware decomposition (FADD) for degradation discovery → precise adaptation.



✅ \*\*Mixture-of-Experts Network\*\*  

Specialized experts (dehazing, denoising, deblurring, enhancement) with probabilistic routing and sparse activation.



✅ \*\*Physically Consistent Optimization\*\*  

Degradation-Aware Optimal Transport (DAOT) loss ensures statistical–geometric fidelity across degradations.



---



\## 📊 Quantitative Results



| Dataset | Dehazing (PSNR↑) | Deblurring | Denoising | Low-Light |

|----------|------------------|-------------|------------|------------|

| \*\*MD-RSID\*\* | \*\*26.86\*\* | \*\*27.73\*\* | \*\*32.77\*\* | \*\*31.96\*\* |

| \*\*MD-RRSHID\*\* | \*\*22.96\*\* | \*\*33.73\*\* | \*\*35.17\*\* | \*\*37.35\*\* |

| \*\*MDRS-Landsat\*\* | \*\*39.12\*\* | \*\*36.88\*\* | \*\*34.53\*\* | \*\*42.24\*\* |



PhyDAE consistently surpasses \*\*10+ state-of-the-art all-in-one restoration models\*\*, achieving both \*\*superior performance\*\* and \*\*remarkable efficiency gains\*\*.



<p align="center">

&nbsp; <img src="assets/vis\_results.png" width="95%">

</p>



---



\## 🔧 Installation



```bash

\# Clone the repository

git clone https://github.com/HIT-SIRS/PhyDAE.git

cd PhyDAE



\# Create environment

conda create -n phydae python=3.9

conda activate phydae



\# Install dependencies

pip install -r requirements.txt



