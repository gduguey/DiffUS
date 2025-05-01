# DiffUS - Differentiable Ultrasound from MRI

**Authors:** Noé Bertramo, Gabriel Duguey  
**Institution:** Massachusetts Institute of Technology (MIT)  
**Date:** Spring 2025  
**Dataset:** [ReMIND2Reg](https://doi.org/10.7937/3RAG-D070)  
**Keywords:** Fast Differentiable Simulation · Ultrasound Imaging · MRI · Registration · Alignement

---

This project investigates the **differentiable simulation of 2D ultrasound images from 3D MRI volumes** by modeling wave propagation through soft tissues.  
The goal is to learn a differentiable forward model of ultrasound generation that allows **training on paired MRI/US data** to eventually **recover the source position** or other acquisition parameters through backpropagation.

---

## Paths

[Acoustics Physics](USPhysics.md)
[Forward Modeling](forward_physics.md)

## File Naming Convention in the ReMIND2Reg dataset

| File suffix       | Modality           | When acquired    | Purpose                          |
|-------------------|--------------------|------------------|----------------------------------|
| `_0000.nii.gz`    | 3D ultrasound (iUS) | During surgery   | After tumor resection            |
| `_0001.nii.gz`    | MRI - ceT1          | Before surgery   | Contrast-enhanced, structural    |
| `_0002.nii.gz`    | MRI - T2-SPACE      | Before surgery   | Non-contrast, structural         |

Each file is a **3D matrix of size `256×256×256`** representing the brain volume in 3D space, with **0.5 mm voxel spacing**.


## Useful Resources

- [**Preoperative-to-Intraoperative Brain Image Registration Paper**](https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=xdECLMkAAAAJ&citation_for_view=xdECLMkAAAAJ:7PzlFSSx8tAC):  
  Reuben Dorent's paper introduces the ReMIND2Reg dataset and benchmarks for brain image registration between MRI and intra-operative ultrasound.

- [**ReMIND Dataset on The Cancer Imaging Archive (TCIA)**](https://www.cancerimagingarchive.net/collection/remind/):  
  Official data collection page for ReMIND, containing pre- and intra-operative brain tumor imaging from 114 patients.

- [**ReMIND2Reg 2024 Zenodo Release**](https://zenodo.org/records/12700312):  
  Download page for the 2024 version of the ReMIND2Reg dataset, including training and validation data in NIfTI format.

- [**DiffDRR Visualization Code**](https://github.com/eigenvivek/DiffDRR/blob/main/diffdrr/visualization.py):  
Vivek Gopalakrishnan's python script from the DiffDRR repo that handles rendering and visualization of differentiable X-ray projections, which can be adapted for ultrasound simulation. --> More on [his personal website](https://vivekg.dev/).
