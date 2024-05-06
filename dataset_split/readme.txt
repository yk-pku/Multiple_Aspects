original:
The Left Atrial (LA) dataset: The LA dataset was obtained from the 2018 Left Atrium Segmentation Challenge , 
and it includes 100 3D gadolinium-enhanced MRI scans collected by either a 1.5 Tesla Avanto or 3.0 Tesla Verio clinical whole-body scanner, 
from patients with AF prior to 3 to 27 months post clinical ablation from the University of Utah. 
As provided by the competition organizing committee, the spatial resolution of a 3D MRI scan was 0.625×0.625×0.625 mm3 with spatial dimensions of either 576×576×88 or 640×640×88 pixels. 
The LA cavity was defined as the pixels contained within the LA endocardial surface (including the mitral valve, LA appendage and an extent of the pulmonary vein (PV) sleeves), which were manually segmented in consensus with three trained observers for each LGE-MRI scan using the Corview image processing software(Merrk Inc, Salt Lake City, UT) (McGann et al. 2014). 
The grayscale LGE-MRI image volumes and associated binary LA segmentations were stored in the nearly-raw raster data (nrrd) format.

processed:
For LA dataset, the raw images were converted to 3-channel PNG format, 
whose spatial resolution were 576×576 or 640×640 and Z direction depth (slice num) is less than 88(remove some slices).
XXX/lgemri.nrrd means images
XXX/laendo.nrrd means binary annotations
