
```
Include different sets of parameters that we used for biological system simulation
    
The first set of parameter "global_parameter_1" is set up depending on the heatmap 
corresponding to higher eigenvalue of PDE system


The second set of parameter "global_parameter_2" is depending on the parameter "A closed NPZ model with delayed nutrient recycling"
with link: https://link.springer.com/article/10.1007/s00285-013-0646-x/tables/1

```
struct GlobalParameter
    Nₜ::Vector{Float64}
    λ::Float64
    ν::Float64
    δ::Float64
    g::Float64
    m::Float64
end

struct DiffusionParameter
    d₁::Float64
    d₂::Float64
    d₃::Float64
end

struct WaveNumber
    k::Float64
end

struct Diffusion_Vec_Parameter
    d_vec::Vector{Float64}
end

struct Total_Population
    total::Float64
end



## Define the set of the global parameters:
global_parameter_1 = GlobalParameter(0.1:0.1:2, 0.017, 5.9, 0.17, 7, 1/2)

global_parameter_2 = GlobalParameter(1.75:0.05:3, 0.85, 2, 1, 2, 1/2)




