## SECTION 1 : Multimodal Ceramic Glaze Retrieval Engine

<img width="665" height="707" alt="image" src="https://github.com/user-attachments/assets/24e7f191-77eb-40dc-a47b-d29d95f65d17" />


### Project Background & Motivation
Ceramic glazes are complex glass formulations that fuse to clay bodies during firing, providing waterproofing, structural durability, and essential aesthetic dimensions. Because glazes rely on countless combinations of glass formers, fluxes, stabilizers, colorants, and crystallizers, empirical testing and formulation require extensive domain expertise and trial-and-error—a hurdle that often deters practitioners.   While digital repositories like Glazy and Digitalfire provide access to tens of thousands of open-source recipes, existing retrieval methods depend heavily on keyword Boolean matching or exact stoichiometric inputs. These traditional query mechanisms fail to capture the multi-dimensional, visual behavior of glazes as perceived by ceramicists. Furthermore, industry supply chain disruptions—such as the sudden discontinuation of popular commercial lines due to raw material shortages—make finding reliable, accessible alternative glaze formulations a critical necessity.

### Project Overview & Executive Summary
To bridge the gap between a ceramicist's intended visual aesthetic and available online recipes, the Multimodal Ceramic Glaze Retrieval Engine implements an end-to-end multi-modal data fusion architecture. The system maps low-level visual inputs (such as photographs of pottery test tiles or finished vessels) to quantitative glaze appearances, chemical compositions, and behavioral traits.

The engine's core capabilities include:    
**Automated Image Segmentation & Object Extraction:** Isolates ceramic objects and test tiles from complex photographic backgrounds using Grounded-SAM and OpenCLIP pipelines.   
**Multimodal Feature Extraction & Shared Latent Space:** Aligns chemical oxide compositions (processed via PCA and trainable projection heads) with descriptive text attributes (processed via MiniLM) using a symmetric InfoNCE loss function.      
**Perceptual Color & Metadata Reranking:** Integrates CIEDE2000 perceptual color matching, surface finish classification, and firing cone compatibility to score and rank recommendations.   
**Cluster-Based Result Diversification:** Utilizes $k$-means chemical composition clustering to help users explore diverse chemical formulations that achieve similar fired appearances.     
**Dual-Interface Accessibility:** Provides both a local interactive Streamlit web application (glaze_retrieval_app.py) and a Jupyter Notebook workflow (`glaze_retrieval.ipynb`) for flexible deployment and experimentation.     

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
Ceramic glaze formulation is traditionally governed by exhaustive empirical testing and domain expertise, while existing digital repositories remain restricted to rigid keyword or stoichiometric queries that fail to capture nuanced visual behavior. To address this gap, this paper introduces the Multimodal Ceramic Glaze Retrieval Engine, an end-to-end framework that maps low-level visual inputs to quantitative glaze appearances and chemical compositions. The system automates object extraction and segmentation using Grounded-SAM and OpenCLIP pipelines, and aligns oxide chemical compositions with descriptive text features within a shared latent space optimized via InfoNCE loss. Retrieval results are scored and reranked using a hybrid weighting scheme combining shared latent similarity, CIEDE2000 perceptual color matching, surface finish compatibility, and firing cone requirements, alongside $k$-means chemical clustering for result diversification. Evaluated across real-world ceramic photographs, the system achieves a 95% Hit@5 and a mean Precision@5 of 56%, demonstrating significant improvements in bridging the divide between visual aesthetics and chemical recipe discovery. 

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Lim Hao | A0354214M | project and technical research, system architecture, model training | zdino1998@gmail.com |
| Pang Claire Fuiyi | A0357738L | system architecture, data processing, model training, retrieval algorithm, front-end app | clairepang22@gmail.com |
| Polthai Nakhin | A0355086X | system architecture, model training, evaluation metrics | nakhinbill@gmail.com |
| Le Duc Nghia | A0356261E | image segmentation algorithm, color extraction pipeline, front-end app | nghialeduc268@gmail.com |

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO
https://www.youtube.com/watch?v=i5HCFqvJWp0

## SECTION 5 : USER GUIDE

See the `README.md` in [glaze_retrieval](glaze_retrieval/) to get create a Python virtual environment and start the Streamlit app or run the notebook.

| Folder | What it holds | 
| --- | --- | 
| `training/` | Scripts used to train the latent space, projection heads,
| `training/dataset_creation` | Data processing scripts and intermediate datasets to create training dataset
| `glaze_retrieval/` | Streamlit app and Jupyter notebook for uploading image and running retrieval

## SECTION 6 : PROJECT REPORT / PAPER
