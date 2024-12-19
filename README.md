# PSO + SA Image Segmentation
- Empowering traditional clustering techniques with evolutionary algorithms, here two strong ones, namely particle swarm optimization and simulated annealing are used.
- Enjoy!!!
  
![PSO + SA Image Segmentation](https://user-images.githubusercontent.com/11339420/152088483-217a73fd-757e-46c8-a25d-a376f8a14032.jpg)


This repository contains an implementation of Particle Swarm Optimization (PSO) and Simulated Annealing (SA) for image segmentation using clustering techniques. The hybrid approach refines traditional clustering methods with evolutionary algorithms for improved segmentation performance.

## Features
- **Particle Swarm Optimization (PSO)** for finding optimal cluster centers.
- **Simulated Annealing (SA)** for refining the results of PSO.
- **Dynamic Visualizations**: Plots the costs of PSO and SA iterations over time.
- **Image Segmentation**: Segments images based on pixel intensity and RGB channels.
- **Median Filtering**: Enhances the segmented output for better visual quality.



## Parameters
You can customize the following parameters in the script:

- **PSO Parameters**:
  - `swarm_size`: Number of particles in the swarm (default: 250).
  - `iterations`: Number of iterations for PSO (default: 50).
  - `w`: Inertia weight (default: 0.7).
  - `c1`: Personal acceleration coefficient (default: 1.5).
  - `c2`: Global acceleration coefficient (default: 1.5).

- **SA Parameters**:
  - `sa_iterations`: Number of iterations for SA (default: 10).
  - `sa_temp`: Initial temperature for SA (default: 100).
  - `sa_cooling`: Cooling rate for SA (default: 0.9).

- **Clustering**:
  - `clusters`: Number of clusters for segmentation (default: 7).

## Outputs
### Segmented Images
- **Original Image**
- **Grayscale Segmented Image**
- **Color Segmented Image**
- **Median Filtered Grayscale Segmentation**
- **Median Filtered Color Segmentation**

### Cost Evolution
- **PSO Iteration Costs**: Displays the best cost achieved during each PSO iteration.
- **SA Iteration Costs**: Shows the cost refinement during SA iterations.
