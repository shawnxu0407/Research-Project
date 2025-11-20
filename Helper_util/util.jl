```
Include all the helper function to run the 2D/3D analysis for biological PDE/ODE Eigenvalues and Simulation 
```


using LinearAlgebra
using Polynomials
using Oceananigans
using Plots
using FFMPEG
import Random
using Distributions
using NCDatasets
using FFTW
using Polynomials: fit
using MeshGrid
using Peaks
using Roots
using PyPlot
using Colors

```
This First chunk of until functions are for the 2D system simulation
```


## The following function computes the equilibrium points N and P
function equilibrium_state_2D(N_t,λ,ν)
    ## record the length of vector N_t
    num_pt=length(N_t)
    # Pre-define the vector P_star and N_star as vector to store the equilibrium values for each N_t[i]
    P̄,N̄=zeros(Float64,num_pt),zeros(Float64,num_pt)

    # running the for loop for every N_t
    for i in 1:num_pt
       poly_i = Polynomial([λ^2, 2 * λ^2, (λ^2-ν^2 * N_t[i]), ν^2])
       ## Only consider the real roots of above polynomial for solving N
       all_roots = roots(poly_i)
       real_roots = filter(x -> isreal(x), all_roots)
       real_roots = Float64.(real(real_roots))

       ## if there is no such real roots in the restricted range [0,N_t[i]], we use -1 to denote N and P
       if isempty(filter(x -> 0 <= x <= N_t[i], real_roots))
           N̄[i] = -1
           P̄[i] = -1
       else
           N̄[i] = filter(x -> 0 <= x <= N_t[i], real_roots)[1]
           P̄[i] = N_t[i] - N̄[i]
       end
   end
   
   return(N̄,P̄)
end

## ODE_PDE_system returns a list with 1st and 2nd column denotes the maximum real part of eigenvalue of ODE and PDE 
## 3rd column of their corresponding N_t position
## 4th column returns the eigen_vector with the eigenvalue of the maximum real part


function ODE_PDE_system_2D(Nₜ,λ,ν,m,d₁,d₂,k)
    N̄,P̄=equilibrium_state_2D(Nₜ,λ,ν)
    result=[]
    for i in 1:length(Nₜ)
        Nᵢ,Pᵢ=N̄[i],P̄[i]
        if Nᵢ == -1 && Pᵢ==-1
           key=[-0.1,-0.1, Nₜ[i]]
        else
           ## Define the Jacobian matrix at equilibrium states
           A=zeros(2,2)
           A=[-ν*Pᵢ/((1+Nᵢ)^2)   (m-1)*λ*Pᵢ^(m-1)
               ν*Pᵢ/((1+Nᵢ)^2)   -(m-1)*λ*Pᵢ^(m-1)]
           ## define the linear approximation matrix for PDE system
           diffusion_vec=[d₁,d₂]
           diffusion_mat=diagm(0 => -(k^2)*diffusion_vec)
           pde_mat=A+diffusion_mat
           ode_mat=A
           eigen_pde=eigen(pde_mat).values
           ## define the maximum of real part of eigenvalue
           key_eigen_pde = maximum(real(eigen_pde))
           max_index= argmax(real(eigen_pde))

           eigen_ode=eigen(ode_mat).values
           key_eigen_ode = maximum(real(eigen_ode))
           ## define the eigenvector corresponding to the largest eigenvalue
           eigen_vec=(eigen(pde_mat).vectors)[:,max_index]
           
           key=[key_eigen_ode,key_eigen_pde,Nₜ[i],eigen_vec]
           
        end
        push!(result,key)
    end
    return(result)
end



function largest_eigenvalue(d, N_star, P_star)
    k_square = 0:0.1:50
    largest_eigen=zeros(length(k_square))
    for i in 1:length(k_square)
        h = d* k_square[i]^ 2+ (d* 2 * P_star/ ((1 + N_star)^ 2)- 0.7* P_star^(-1/2)* 0.5)* k_square[i]
        b = (d+1)* (k_square[i])- (0.5 * 0.7 * P_star^(-1/2)- 2 * P_star / ((1 + N_star) ^ 2))
        poly = Polynomial([h,b,1])
        roots_poly = roots(poly)
        largest_eigen[i] = maximum(real(roots_poly))
    end
    p=plot(k_square, largest_eigen, label="d = $d", xlabel="k^2", ylabel="Re \\lambda",ylim = (0, 0.4))

    max_eigen = maximum(largest_eigen)
    max_index = argmax(largest_eigen)
    corresponding_k_square = k_square[max_index]
    
    return p,"max_real_eigenvalue" ,max_eigen,"corresponding k value" ,sqrt(corresponding_k_square), "diffusion ratio" ,d
end


## The characteristic poly that we are solving is k^4 + b k^4 + h=0
function find_largest_eigenvalue(d, k, N_star, P_star)
    k_square=k^2
    h = d* k_square^ 2+ (d* 2 * P_star/ ((1 + N_star)^ 2)- 0.7* P_star^(-1/2)* 0.5)* k_square
    b = (d+1)* k_square- (0.5 * 0.7 * P_star^(-1/2)- 2 * P_star / ((1 + N_star) ^ 2))
    poly = Polynomial([h,b,1])
    roots_poly = roots(poly)
    largest_eigen = maximum(real(roots_poly))
    
    return "k value", k , "eigen_value", largest_eigen
end


## key_variable_outcome takes the specific file_writer to returns the actual numeric of N, P, Space and time array
## with time_lim, we can output any N and P data corresponding to this time_limit


using PyCall
@pyimport matplotlib.ticker as ticker
function heatmap_plot_2D(save_img_path, N_data, P_data, times, space)
    N_data_slice = N_data[:, 20:end]
    P_data_slice = P_data[:, 20:end]

    # Also slice times accordingly since you're slicing columns (assuming columns correspond to time)
    times_slice = times[20:end]

    # Determine extent with sliced data
    extent_N = [minimum(times_slice), maximum(times_slice), minimum(space), maximum(space)]
    extent_P = [minimum(times_slice), maximum(times_slice), minimum(space), maximum(space)]

    fig, axs = subplots(1, 2, figsize=(20, 12), constrained_layout=true)

    # Heatmap for N_data with actual axes
    cax1 = axs[1].imshow(N_data_slice, aspect="auto", cmap="inferno", extent=extent_N, origin="lower")
    axs[1].set_title("N", fontsize=16)
    axs[1].set_xlabel("Time", fontsize=14)
    axs[1].set_ylabel("Modes", fontsize=14)

    cbar1 = fig.colorbar(cax1, ax=axs[1], shrink=0.8, aspect=20)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label("", fontsize=14)


    # Heatmap for P_data with actual axes
    cax2 = axs[2].imshow(P_data_slice, aspect="auto", cmap="inferno", extent=extent_P, origin="lower")
    axs[2].set_title("P", fontsize=16)
    axs[2].set_xlabel("Time", fontsize=14)

    

    # Colorbar
    cbar2 = fig.colorbar(cax2, ax=axs[2], shrink=0.8, aspect=20)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label("", fontsize=14)

    # Save figure
    PyPlot.savefig(save_img_path, dpi=300, bbox_inches="tight")
end




## Animation of N
function make_animation_N(N_data,P_data,times,space,time_step)
    
    anim_N = @animate for t in 1:time_step:size(N_data, 2)
        plot(space, N_data[:, t],ylim=(N̄-0.15,N̄+0.15),label="N")
        time=times[t]
        title!("Time: $time")
    end
    
    mp4(anim_N, "D:/Master Research/N_animation.mp4", fps=20)

end
    


## Animation of P
function make_animation_P(N_data,P_data,times,space,time_step)
    anim_P = @animate for t in 1:time_step:size(P_data, 2)
        plot(space, P_data[:, t],ylim=(P̄-0.1,P̄+0.1),label="P")
        time=times[t]
        title!("Time: $time")
    end
    
    mp4(anim_P, "D:/Master Research/P_animation.mp4", fps=20)

end
    

## Conservation plot over time
function conservation_plot(N_data,P_data,times,space)
    ## pre-define a vector to record the total concentration over time
    total_con = sum(N_data, dims=1) .+ sum(P_data, dims=1)
    total_con_vector = vec(total_con)
    # Plot the row sums against the time vector
    plot(times, total_con_vector, xlabel="Time", ylabel="sum con over time", title="total concentration over time",ylim=(180,190),label="total population")
end


## Surface plot
function surface_plot(N_data,P_data,times,space)
    N_plot=surface(times, space, N_data ,xlabel="Time", ylabel="Space", title="N Surface Plot")
    P_plot=surface(times, space, P_data ,xlabel="Time", ylabel="Space", title="P Surface Plot")

    plot(N_plot, P_plot, layout=(1, 2),size=(1200, 800))
end


## E_folding time
function e_folding_time(data_mat_N,data_mat_P,times,space,N_star,P_star)
    time_array_N = Float64[]
    push!(time_array_N,times[1])
    perturb_N=maximum(abs.(data_mat_N[:,1].-N_star))
    
    for t in 2:size(times)[1]
        if maximum(abs.(data_mat_N[:,t].-N_star)) >= exp(1)*perturb_N
           push!(time_array_N,times[t])
           perturb_N=maximum(abs.(data_mat_N[:,t].-N_star))
        end
    end


    time_array_P = Float64[]
    push!(time_array_P,times[1])
    perturb_P=maximum(abs.(data_mat_P[:,1].-P_star))
    
    for t in 2:size(times)[1]
        if maximum(abs.(data_mat_P[:,t].-P_star)) >= exp(1)*perturb_P
           push!(time_array_P,times[t])
           perturb_P=maximum(abs.(data_mat_P[:,t].-P_star))
        end
    end

    time_array_N=diff(time_array_N)
    time_array_P=diff(time_array_P)

    time_axis_N=range(0, stop=length(time_array_N), length=length(time_array_N))
    time_axis_P=range(0, stop=length(time_array_P), length=length(time_array_P))

    e_folding_N=1 ./time_array_N
    e_folding_P=1 ./time_array_P
    
    myplot_N = scatter(time_axis_N, e_folding_N, xlabel="Time", ylabel="Growth Rate", title="E-folding N", label="N", legend=:bottomright, color=:blue)
    plot!(myplot_N, time_axis_N, e_folding_N, label="", line=:solid, color=:blue)
    ylims!(0, maximum(e_folding_N)+0.1)
    
    myplot_P = scatter(time_axis_P, e_folding_P, xlabel="Time", ylabel="Growth Rate", title="E-folding P", label="P", legend=:bottomright, color=:blue)
    plot!(myplot_P, time_axis_P, e_folding_P, label="", line=:solid, color=:red)
    ylims!(0, maximum(e_folding_P)+0.1)

    
    return e_folding_N,e_folding_P,myplot_N,myplot_P
end

function experiment_growth_rate_2D(file_name, N̄, P̄, I₁, I₂)
    ds = NCDataset(file_name, "r")
    times = ds["time"][:]

    ## Denote the perturbation for N and P from the equilibrium state
    N′ = ds["perturbation_N"]
    P′ = ds["perturbation_P"]
    time_increment=170/size(times)[1]

    

    Interval_1=Int(round(I₁[1]/time_increment)):Int(round(I₁[size(I₁)[1]]/time_increment))
    Interval_2=Int(round(I₂[1]/time_increment)):Int(round(I₂[size(I₂)[1]]/time_increment))
    
    degree = 1

    ## Fit the log of growth with line on time range I for N,P
    
    linear_fit_N = fit(times[Interval_1], log.(N′[Interval_1]), degree, var = :t)
    best_fit_N = @. exp(linear_fit_N[0] + linear_fit_N[1] * times)
    
    linear_fit_P = fit(times[Interval_2], log.(P′[Interval_2]), degree, var = :t)
    best_fit_P = @. exp(linear_fit_P[0] + linear_fit_P[1] * times)

    print("Growth rate of N is approximately ", linear_fit_N[1], "\n")
    print("Growth rate of P is approximately ", linear_fit_P[1], "\n")
    print("Largest real part of e-value ", ODE_PDE_system_2D(total_population,λ,ν,m,d₁,d₂,k)[1][2])


    plot(times, N′,label="norm(N′)", yscale = :log10, linestyle=:solid,
    lw=4, xlabel="time", ylabel="norm",title="Norm of perturbations", legend=:left)

    plot!(times, P′,label="norm(P′)", linestyle=:solid, lw=4)#
    
    plot!(times, best_fit_N,label="N best fit", linestyle=:dash, lw=4)
    
    plot!(times, best_fit_P,label="P best fit", linestyle=:dash, lw=4)

end
    
## Heatmap for power of each mode
function FFT_power_2D(save_path, N_data, P_data, N̄, P̄, times)
    perturbation_N = N_data .- N̄
    perturbation_P = P_data .- P̄

    mode_values = (1:41 .- 1) / 2

    rev_N_data=reverse(perturbation_N, dims=1)
    N_data_mat=vcat(rev_N_data, perturbation_N)
    fft_coeff_N_data=zeros(size(N_data_mat))


    rev_P_data=reverse(perturbation_P, dims=1)
    P_data_mat=vcat(rev_P_data, perturbation_P)
    fft_coeff_P_data=zeros(size(P_data_mat))

    for i in 1:size(P_data_mat)[2]
        fft_coeff_P_data[:,i]=abs.( fft(P_data_mat[:,i]) )
        fft_coeff_N_data[:,i]=abs.( fft(N_data_mat[:,i]) )
    end

    mode_range = 2:21
    mode_values = (mode_range .- 1) ./ 2

    # Use your heatmap plotting function
    heatmap_plot_2D(save_path,
        fft_coeff_N_data[mode_range, :],
        fft_coeff_P_data[mode_range, :],
        times,
        mode_values)

end


function key_variable_outcome_2D(mymodel, file_name)
    space=Array(znodes(mymodel.tracers.N))
    N_timeseries = FieldTimeSeries(file_name, "N")
    P_timeseries = FieldTimeSeries(file_name, "P")
    times=Array(N_timeseries.times)

    N_data = parent(N_timeseries.data[:,:,1:size(space)[1],:])
    P_data = parent(P_timeseries.data[:,:,1:size(space)[1],:])
    N_data = dropdims(N_data, dims=(1, 2))
    P_data = dropdims(P_data, dims=(1, 2))
    
    time_index=size(times)[1]

    N_data=N_data[:, 1:time_index]
    P_data=P_data[:, 1:time_index]

    return N_data,P_data,times,space
end

function plot_pde_eigenvalues_2D(Nₜ, λ, ν, m, d₁, d₂_values, k_range, xlim_range, ylim_range)
    p = plot(xlabel="k", ylabel="Max Re(λ)",
             title="m=0.5",
             lw=2,
             xlim=xlim_range,
             ylim=ylim_range)
    
    # Loop over different d₂ values
    for d₂ in d₂_values
        eigen_pde_values = []
        for k in k_range
            result = ODE_PDE_system_2D(Nₜ, λ, ν, m, d₁, d₂, k)[1]
            push!(eigen_pde_values, result[2])
        end
        # Add line to the plot for current d₂
        plot!(p, k_range, eigen_pde_values, label="d₂ = $d₂",lw=2)
    end

    # Add horizontal line at 0
    hline!(p, [0], linestyle=:dash, color=:black, label="")

    return p
end



function plot_PDE_heatmaps_2D(ODE_PDE_system_2D, λ, ν, m, d₁, k_values;
                           Nₜ_range=0.1:0.01:1, d₂_range=0:0.0001:0.004,
                           cmap="bwr", n_xticks=5, n_yticks=5,
                           output_file="multi_heatmap_PDE_eigen.png")

    Nₜ = collect(Nₜ_range)
    d₂_vec = collect(d₂_range)
    

    fig, axes = subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor("white")

    for (idx, k) in enumerate(k_values)
        row = div(idx - 1, 2) + 1
        col = (idx - 1) % 2 + 1
        ax = axes[row, col]

        PDE_eigen = zeros(Float64, length(d₂_vec), length(Nₜ))
        for i in eachindex(d₂_vec)
            results = ODE_PDE_system_2D(Nₜ, λ, ν, m, d₁, d₂_vec[i], k)
            for j in eachindex(Nₜ)
                _, PDE_eigen[i, j], _ = results[j]
            end
        end

        # Diverging colormap centered at 0
        vmax = maximum(abs, PDE_eigen)
        c = ax.imshow(PDE_eigen,
                      extent=[minimum(Nₜ), maximum(Nₜ), minimum(d₂_vec), maximum(d₂_vec)],
                      aspect="auto", origin="lower",
                      cmap=cmap,
                      vmin=-vmax, vmax=vmax)  # center colormap at 0

        # Draw zero contour
        cs=ax.contour(Nₜ, d₂_vec, PDE_eigen, levels=[0.0], colors="black", linewidths=1.5)
        ax.clabel(cs, inline=true, fontsize=10, fmt="0")   # label on contour

        ax.set_title("k = $k")
        ax.set_xlabel(L"N_{T}")
        ax.set_ylabel(L"d_{2}")
        ax.tick_params(direction="out")
        ax.set_xticks(range(minimum(Nₜ), stop=maximum(Nₜ), length=n_xticks))
        ax.set_yticks(range(minimum(d₂_vec), stop=maximum(d₂_vec), length=n_yticks))

        fig.colorbar(c, ax=ax)
    end

    tight_layout()
    PyPlot.savefig(output_file, dpi=300, facecolor=fig.get_facecolor())
    close(fig)
end



```
The following untility function are for the 3D system simulation
```

function equilibrium_state_3D(N_t, λ, ν, δ, g, m)
    ## record the length of vector N_t
    num_pt=length(N_t)
    # Pre-define the vector P_star and N_star as vector to store the equilibrium values for each N_t[i]
    P̄, N̄, Z̄=zeros(Float64,num_pt), zeros(Float64,num_pt), zeros(Float64,num_pt)

    # running the for loop for every N_t
    for i in 1:num_pt
        
       b=1+ (ν+δ)/(g-δ) - (λ/δ)*(δ/(g-δ))^m - N_t[i]
       c=  δ/(g-δ) - (λ/δ)*(δ/(g-δ))^m -N_t[i]

       poly_i = Polynomial([c, b, 1])
       ## Only consider the real roots of above polynomial for solving N
       all_roots = roots(poly_i)
       real_roots = filter(x -> isreal(x), all_roots)
       real_roots = Float64.(real(real_roots))

       ## if there is no such real roots in the restricted range [0,N_t[i]], we use -1 to denote N and P
       if isempty(filter(x -> 0 <= x <= N_t[i], real_roots))
           N̄[i] = -1
           P̄[i] = -1
           Z̄[i] = -1
       else
           N̄[i] = filter(x -> 0 <= x <= N_t[i], real_roots)[1]
           P̄[i] = δ/(g-δ)
           Z̄[i] = N_t[i] - N̄[i] - P̄[i]
       end
       
       ## make a double check that all N P Z are in the range of [0, N_t]
       if !(0 <= N̄[i] <= N_t[i] && 0 <= P̄[i] <= N_t[i] && 0 <= Z̄[i] <= N_t[i])
        N̄[i], P̄[i], Z̄[i] = -1, -1, -1
       end
   end
   
   return(N̄,P̄,Z̄)
end



## Change -0.1 to NAN values

function ODE_PDE_system_3D(Nₜ, λ, ν, δ, g, m, d₁, d₂, d₃, k)
    N̄,P̄,Z̄=equilibrium_state_3D(Nₜ, λ, ν, δ, g, m)
    result=[]
    for i in 1:length(Nₜ)
        Nᵢ,Pᵢ,Zᵢ=N̄[i],P̄[i],Z̄[i]
        if Nᵢ == -1 && Pᵢ==-1 && Zᵢ==-1
           key=[NaN, NaN, Nₜ[i]]
        else
           ## Define the Jacobian matrix at equilibrium states
           A=zeros(3,3)
           A=[-ν*Pᵢ/((1+Nᵢ)^2)   -ν*(Nᵢ/(Nᵢ+1))+ m*λ*Pᵢ^(m-1)    δ
              ν*Pᵢ/((1+Nᵢ)^2)   ν*(Nᵢ/(Nᵢ+1))-g*Zᵢ/((1+Pᵢ)^2)-m*λ*Pᵢ^(m-1)  -g*Pᵢ/(1+Pᵢ)
              0                 g*Zᵢ/((1+Pᵢ)^2)      0]
           ## define the linear approximation matrix for PDE system
           diffusion_vec=[d₁,d₂,d₃]
           diffusion_mat=diagm(0 => -(k^2)*diffusion_vec)
           pde_mat=A+diffusion_mat
           ode_mat=A
           eigen_pde=eigen(pde_mat).values
           ## define the maximum of real part of eigenvalue
           key_eigen_pde = maximum(real(eigen_pde))
           max_index= argmax(real(eigen_pde))

           eigen_ode=eigen(ode_mat).values
           key_eigen_ode = maximum(real(eigen_ode))
           ## define the eigenvector corresponding to the largest eigenvalue
           eigen_vec=(eigen(pde_mat).vectors)[:,max_index]
           eigen_val=(eigen(pde_mat).values)[max_index]
           
           key=[key_eigen_ode, key_eigen_pde, Nₜ[i], eigen_vec, eigen_val]
           
         end
         push!(result,key)
     end
return(result)
end







function key_variable_outcome_3D(time_lim, mymodel, file_name)
    space=Array(znodes(mymodel.tracers.N))
    N_timeseries = FieldTimeSeries(file_name, "N")
    P_timeseries = FieldTimeSeries(file_name, "P")
    Z_timeseries = FieldTimeSeries(file_name, "Z")
    times=Array(N_timeseries.times)

    N_data = parent(N_timeseries.data[:,:,1:size(space)[1],:])
    P_data = parent(P_timeseries.data[:,:,1:size(space)[1],:])
    Z_data = parent(Z_timeseries.data[:,:,1:size(space)[1],:])
    N_data = dropdims(N_data, dims=(1, 2))
    P_data = dropdims(P_data, dims=(1, 2))
    Z_data = dropdims(Z_data, dims=(1, 2))
    
    @assert time_lim <= times[length(times)] "time_lim can not be larger than the simulated time"
    times=times[times.<time_lim]
    time_index=size(times)[1]

    N_data=N_data[:, 1:time_index]
    P_data=P_data[:, 1:time_index]
    Z_data=Z_data[:, 1:time_index]

    return N_data,P_data,Z_data,times,space
end



## Surface plot
function surface_plot_3D(N_data,P_data,Z_data,times,space)
    N_plot=surface(times, space, N_data ,xlabel="Time", ylabel="Space", title="N Surface Plot")
    P_plot=surface(times, space, P_data ,xlabel="Time", ylabel="Space", title="P Surface Plot")
    Z_plot=surface(times, space, Z_data ,xlabel="Time", ylabel="Space", title="Z Surface Plot")

    plot(N_plot, P_plot, Z_plot, layout=(3,1),size=(1200, 800))
end


## The characteristic poly that we are solving is k^4 + b k^4 + h=0
function find_largest_eigenvalue(d, k, N_star, P_star)


    
    return "k value", k , "eigen_value", largest_eigen
end


## Conservation plot over time
function conservation_plot_3D(N_data,P_data,Z_data,times,space)
    ## pre-define a vector to record the total concentration over time
    total_con = sum(N_data, dims=1) .+ sum(P_data, dims=1) .+ sum(Z_data,dims=1)
    total_con_vector = vec(total_con)
    # Plot the row sums against the time vector
    plot(times, total_con_vector, xlabel="Time", ylabel="sum con over time", title="total concentration over time",ylim=(255,260),label="total population")
end

function plot_eigenvalue_heatmap_3D(save_path, Nₜ, λ, ν, δ, g, d₁, d₂, d₃, m_range, k_range; cmap="coolwarm")
    heatmap_data = zeros(length(m_range), length(k_range))

    for (i, m) in enumerate(m_range)
        for (j, k) in enumerate(k_range)
            result = ODE_PDE_system_3D(Nₜ, λ, ν, δ, g, m, d₁, d₂, d₃, k)[1]
            heatmap_data[i, j] = real(result[2])
        end
    end

    fig, ax = subplots()

    # automatic symmetric scaling
    vmax = maximum(abs, heatmap_data)
    c = ax.imshow(
        heatmap_data,
        extent=[minimum(k_range), maximum(k_range),
                minimum(m_range), maximum(m_range)],
        aspect="auto", origin="lower",
        cmap=cmap,
        vmin=-vmax, vmax=vmax
    )

    
    cs = ax.contour(
        k_range, m_range, heatmap_data,
        levels=[0.0],
        colors="black",
        linewidths=1.6
    )
    ax.clabel(cs, inline=true, fontsize=10, fmt="0")   # label on contour

    ax.set_xlabel("k")
    ax.set_ylabel("m")
    ax.tick_params(direction="out")
    ax.set_xticks(range(minimum(k_range), stop=maximum(k_range), length=5))
    ax.set_yticks(range(minimum(m_range), stop=maximum(m_range), length=5))

    fig.colorbar(c, ax=ax)

    tight_layout()
    PyPlot.savefig(save_path, dpi=300)
end


function plot_combined_heatmaps_3D(ODE_PDE_system_3D, k_values, Nₜ, λ, ν, δ, g, m, d₁_vec, d₂, d₃;
                                   layout=(2,2), cmap="bwr", n_xticks=5, n_yticks=5,
                                   output_file="3D_combined.png")

    n_subplots = length(k_values)
    nrows, ncols = layout
    Nₜ=collect(Nₜ)
    d₁_vec=collect(d₁_vec)


    fig, axes = subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor("white")


    for (idx, k) in enumerate(k_values)
        row = div(idx - 1, 2) + 1
        col = (idx - 1) % 2 + 1
        ax = axes[row, col]
        # Initialize PDE eigenvalues matrix
        PDE_eigen = zeros(Float64, length(d₁_vec), length(Nₜ))

        # Compute eigenvalues
        for i in 1:length(d₁_vec)
            for j in 1:length(Nₜ)
                _, PDE_eigen[i,j], _ = ODE_PDE_system_3D(Nₜ, λ, ν, δ, g, m, d₁_vec[i], d₂, d₃, k)[j]
            end
        end

        
        # Diverging colormap centered at 0
        vmax = maximum(abs, PDE_eigen)
        c = ax.imshow(PDE_eigen,
                      extent=[minimum(Nₜ), maximum(Nₜ), minimum(d₁_vec), maximum(d₁_vec)],
                      aspect="auto", origin="lower",
                      cmap=cmap,
                      vmin=-vmax, vmax=vmax)  # center colormap at 0

        # Draw zero contour
        cs=ax.contour(Nₜ, d₁_vec, PDE_eigen, levels=[0.0], colors="black", linewidths=1.5)
        ax.clabel(cs, inline=true, fontsize=10, fmt="0")   # label on contour

        ax.set_title("k = $k")
        ax.set_xlabel(L"N_{T}")
        ax.set_ylabel(L"d_{1}")
        ax.tick_params(direction="out")
        ax.set_xticks(range(minimum(Nₜ), stop=maximum(Nₜ), length=n_xticks))
        ax.set_yticks(range(minimum(d₁_vec), stop=maximum(d₁_vec), length=n_yticks))

        fig.colorbar(c, ax=ax)
    end

    # Turn off unused axes

    tight_layout()
    PyPlot.savefig(output_file, dpi=300, facecolor=fig[:get_facecolor]())
    close(fig)
    println("Saved combined 3D heatmap to $output_file")
end



function experiment_growth_rate_3D(file_name,total_population, I₁, I₂, I₃, λ, ν, δ, g, m, d₁, d₂, d₃, k)
    ds = NCDataset(file_name, "r")
    times = ds["time"][:]

    ## Denote the perturbation for N and P from the equilibrium state
    N′ = ds["perturbation_N"]
    P′ = ds["perturbation_P"]
    Z′ = ds["perturbation_Z"]
    time_increment=170/size(times)[1]

    

    Interval_1=Int(round(I₁[1]/time_increment)):Int(round(I₁[size(I₁)[1]]/time_increment))
    Interval_2=Int(round(I₂[1]/time_increment)):Int(round(I₂[size(I₂)[1]]/time_increment))
    Interval_3=Int(round(I₃[1]/time_increment)):Int(round(I₃[size(I₃)[1]]/time_increment))


    target_N=N′[Interval_1]
    N_peaks=findmaxima(target_N)
    target_P=P′[Interval_2]
    P_peaks=findmaxima(target_P)
    target_Z=Z′[Interval_3]
    Z_peaks=findmaxima(target_Z)
    
    degree = 1

    ## Fit the log of growth with line on time range I for N,P
    
    linear_fit_N = fit(times[Interval_1][N_peaks.indices], log.(N′[Interval_1][N_peaks.indices]), degree, var = :t)
    best_fit_N = @. exp(linear_fit_N[0] + linear_fit_N[1] * times)
    
    linear_fit_P = fit(times[Interval_2][P_peaks.indices], log.(P′[Interval_2][P_peaks.indices]), degree, var = :t)
    best_fit_P = @. exp(linear_fit_P[0] + linear_fit_P[1] * times)

    linear_fit_Z = fit(times[Interval_3][Z_peaks.indices], log.(Z′[Interval_3][Z_peaks.indices]), degree, var = :t)
    best_fit_Z = @. exp(linear_fit_Z[0] + linear_fit_Z[1] * times)



    # ODE_PDE_system_3D(total_population, λ, ν, δ, g, m, d₁, d₂, d₃, k)[1][2]

    print("Growth rate of N is approximately ", linear_fit_N[1]," with oscilation period ", 2*mean(diff(times[Interval_1][N_peaks.indices])) , "\n")
    print("Growth rate of P is approximately ", linear_fit_P[1]," with oscilation period ", 2*mean(diff(times[Interval_2][P_peaks.indices])) , "\n")
    print("Growth rate of Z is approximately ", linear_fit_Z[1]," with oscilation period ", 2*mean(diff(times[Interval_3][Z_peaks.indices])) , "\n")
    print("Largest real part of e-value ", ODE_PDE_system_3D(total_population, λ, ν, δ, g, m, d₁, d₂, d₃, k)[1][2])

    


    plot(times, N′,label="norm(N′)", yscale = :log10, linestyle=:solid,
    lw=4, xlabel="time", ylabel="norm",title="Norm of perturbations", legend=:topleft)


    plot!(times, P′,label="norm(P′)", linestyle=:solid, lw=4)#

    plot!(times, Z′,label="norm(Z′)", linestyle=:solid, lw=4)#
    
    plot!(times[Interval_1], best_fit_N[Interval_1],label="N best fit", linestyle=:dash, lw=6)
    
    plot!(times[Interval_2], best_fit_P[Interval_2],label="P best fit", linestyle=:dash, lw=6)

    plot!(times[Interval_3], best_fit_Z[Interval_3],label="Z best fit", linestyle=:dash, lw=6)

end

function plot_pde_eigenvalues_3D(Nₜ, λ, ν, δ, g, m, d₁, d₂, d₃, k_range, xlim_range, ylim_range)
    eigen_pde_values = []
    
    for k in k_range
        result = ODE_PDE_system_3D(Nₜ, λ, ν, δ, g, m, d₁, d₂, d₃, k)[1]
        push!(eigen_pde_values, result[2])
    end

    # Plot max real part of PDE eigenvalue vs k
    p = plot(
        k_range,
        eigen_pde_values,
        xlabel = "k",
        ylabel = "Max Re(λ)",
        title = "m=0.5",
        lw = 2,
        label = "Max Re(λ)",
        ylim = ylim_range,
        xlim = xlim_range
    )
    hline!([0], linestyle=:dash, color=:black, label="")
end




## Get the graph of Equilibrium points for N, P and Z for variable total population


# Function to plot the equilibrium states
function plot_equilibrium_3D(N_t, λ, ν, δ, g, m)

    N̄, P̄, Z̄= equilibrium_state_3D(N_t, λ, ν, δ, g, m)
    computational_sum=N̄+P̄+Z̄

    plot(N_t, N̄, label="N", lw=2)
    plot!(N_t, P̄, label="P", lw=2)
    plot!(N_t, Z̄, label="Z", lw=2)
    plot!(N_t,computational_sum,label="actual sum", linestyle=:dash)
    xlabel!("total population")
    ylabel!("Equilibrium Values")
    title!("Equilibrium Plot")
end


## E_folding time
function e_folding_time_3D(data_mat_N, data_mat_P, data_mat_Z, times, space, N̄, P̄, Z̄)
    
    time_array_N = Float64[]
    push!(time_array_N,times[1])
    perturb_N=maximum(abs.(data_mat_N[:,1].-N̄))
    
    for t in 2:size(times)[1]
        if maximum(abs.(data_mat_N[:,t].-N̄)) >= exp(1)*perturb_N
           push!(time_array_N,times[t])
           perturb_N=maximum(abs.(data_mat_N[:,t].-N̄))
        end
    end


    time_array_P = Float64[]
    push!(time_array_P,times[1])
    perturb_P=maximum(abs.(data_mat_P[:,1].-P̄))
    
    for t in 2:size(times)[1]
        if maximum(abs.(data_mat_P[:,t].-P̄)) >= exp(1)*perturb_P
           push!(time_array_P,times[t])
           perturb_P=maximum(abs.(data_mat_P[:,t].-P̄))
        end
    end


    time_array_Z = Float64[]
    push!(time_array_Z,times[1])
    perturb_Z=maximum(abs.(data_mat_Z[:,1].-Z̄))
    
    for t in 2:size(times)[1]
        if maximum(abs.(data_mat_Z[:,t].-Z̄)) >= exp(1)*perturb_Z
           push!(time_array_Z,times[t])
           perturb_P=maximum(abs.(data_mat_Z[:,t].-Z̄))
        end
    end

    time_array_N=diff(time_array_N)
    time_array_P=diff(time_array_P)
    time_array_Z=diff(time_array_Z)

    time_axis_N=range(0, stop=length(time_array_N), length=length(time_array_N))
    time_axis_P=range(0, stop=length(time_array_P), length=length(time_array_P))
    time_axis_Z=range(0, stop=length(time_array_Z), length=length(time_array_Z))

    e_folding_N=1 ./time_array_N
    e_folding_P=1 ./time_array_P
    e_folding_Z=1 ./time_array_Z
    
    myplot_N = scatter(time_axis_N, e_folding_N, xlabel="Time", ylabel="Growth Rate", title="E-folding N", label="N", legend=:bottomright, color=:blue)
    plot!(myplot_N, time_axis_N, e_folding_N, label="", line=:solid, color=:blue)
    ylims!(0, maximum(e_folding_N)+0.1)
    
    myplot_P = scatter(time_axis_P, e_folding_P, xlabel="Time", ylabel="Growth Rate", title="E-folding P", label="P", legend=:bottomright, color=:blue)
    plot!(myplot_P, time_axis_P, e_folding_P, label="", line=:solid, color=:red)
    ylims!(0, maximum(e_folding_P)+0.1)

    myplot_Z = scatter(time_axis_Z, e_folding_Z, xlabel="Time", ylabel="Growth Rate", title="E-folding Z", label="Z", legend=:bottomright, color=:blue)
    plot!(myplot_Z, time_axis_Z, e_folding_Z, label="", line=:solid, color=:red)
    ylims!(0, maximum(e_folding_Z)+0.1)

    
    return e_folding_N,e_folding_P,e_folding_Z,myplot_N,myplot_P,myplot_Z
end

function heatmap_plot_3D(save_img_path, N_data, P_data, Z_data, times, space)
    N_data_slice = N_data[:, 20:end]
    P_data_slice = P_data[:, 20:end]
    Z_data_slice = Z_data[:, 20:end]

    # Also slice times accordingly since you're slicing columns (assuming columns correspond to time)
    times_slice = times[20:end]

    # Determine extent with sliced data
    extent_N = [minimum(times_slice), maximum(times_slice), minimum(space), maximum(space)]
    extent_P = [minimum(times_slice), maximum(times_slice), minimum(space), maximum(space)]
    extent_Z = [minimum(times_slice), maximum(times_slice), minimum(space), maximum(space)]

    fig, axs = subplots(1, 3, figsize=(15, 6), constrained_layout=true)

    # Heatmap for N_data with actual axes
    cax1 = axs[1].imshow(N_data_slice, aspect="auto", cmap="inferno", extent=extent_N, origin="lower")
    axs[1].set_title("N", fontsize=16)
    axs[1].set_xlabel("Time", fontsize=14)
    axs[1].set_ylabel("Space", fontsize=14)

    cbar1 = fig.colorbar(cax1, ax=axs[1], shrink=0.8, aspect=20)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label("", fontsize=14)

    # Heatmap for P_data with actual axes
    cax2 = axs[2].imshow(P_data_slice, aspect="auto", cmap="inferno", extent=extent_P, origin="lower")
    axs[2].set_title("P", fontsize=16)
    axs[2].set_xlabel("Time", fontsize=14)

    # Colorbar
    cbar2 = fig.colorbar(cax2, ax=axs[2], shrink=0.8, aspect=20)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label("", fontsize=14)

    # Heatmap for P_data with actual axes
    cax3 = axs[3].imshow(Z_data_slice, aspect="auto", cmap="inferno", extent=extent_Z, origin="lower")
    axs[3].set_title("Z", fontsize=16)
    axs[3].set_xlabel("Time", fontsize=14)

    # Colorbar
    cbar3 = fig.colorbar(cax3, ax=axs[3], shrink=0.8, aspect=20)
    cbar3.ax.tick_params(labelsize=12)
    cbar3.set_label("", fontsize=14)

    # Save figure
    PyPlot.savefig(save_img_path, dpi=300, bbox_inches="tight")
end

## Heatmap for power of each mode
function FFT_power_3D(save_path, N_data, P_data, Z_data, N̄, P̄, Z̄, times)
    perturbation_N = N_data .- N̄
    perturbation_P = P_data .- P̄
    perturbation_Z = Z_data .- Z̄

    mode_values = (1:41 .- 1) / 2

    rev_N_data=reverse(perturbation_N, dims=1)
    N_data_mat=vcat(rev_N_data, perturbation_N)
    fft_coeff_N_data=zeros(size(N_data_mat))


    rev_P_data=reverse(perturbation_P, dims=1)
    P_data_mat=vcat(rev_P_data, perturbation_P)
    fft_coeff_P_data=zeros(size(P_data_mat))


    rev_Z_data=reverse(perturbation_Z, dims=1)
    Z_data_mat=vcat(rev_Z_data, perturbation_Z)
    fft_coeff_Z_data=zeros(size(Z_data_mat))

    for i in 1:size(P_data_mat)[2]
        fft_coeff_P_data[:,i]=abs.( fft(P_data_mat[:,i]) )
        fft_coeff_N_data[:,i]=abs.( fft(N_data_mat[:,i]) )
        fft_coeff_Z_data[:,i]=abs.( fft(Z_data_mat[:,i]) )
    end

    mode_range = 2:41
    mode_values = (mode_range .- 1) ./ 2

    # Use your heatmap plotting function
    heatmap_plot_3D(save_path,
        fft_coeff_N_data[mode_range, :],
        fft_coeff_P_data[mode_range, :],
        fft_coeff_Z_data[mode_range, :],
        times,
        mode_values)

end
function experiment_growth_rate_3D(file_name,total_population, I₁, I₂, I₃, λ, ν, δ, g, m, d₁, d₂, d₃, k)
    ds = NCDataset(file_name, "r")
    times = ds["time"][:]

    ## Denote the perturbation for N and P from the equilibrium state
    N′ = ds["perturbation_N"]
    P′ = ds["perturbation_P"]
    Z′ = ds["perturbation_Z"]
    time_increment=170/size(times)[1]

    

    Interval_1=Int(round(I₁[1]/time_increment)):Int(round(I₁[size(I₁)[1]]/time_increment))
    Interval_2=Int(round(I₂[1]/time_increment)):Int(round(I₂[size(I₂)[1]]/time_increment))
    Interval_3=Int(round(I₃[1]/time_increment)):Int(round(I₃[size(I₃)[1]]/time_increment))


    target_N=N′[Interval_1]
    N_peaks=findmaxima(target_N)
    target_P=P′[Interval_2]
    P_peaks=findmaxima(target_P)
    target_Z=Z′[Interval_3]
    Z_peaks=findmaxima(target_Z)
    
    degree = 1

    ## Fit the log of growth with line on time range I for N,P
    
    linear_fit_N = fit(times[Interval_1][N_peaks.indices], log.(N′[Interval_1][N_peaks.indices]), degree, var = :t)
    best_fit_N = @. exp(linear_fit_N[0] + linear_fit_N[1] * times)
    
    linear_fit_P = fit(times[Interval_2][P_peaks.indices], log.(P′[Interval_2][P_peaks.indices]), degree, var = :t)
    best_fit_P = @. exp(linear_fit_P[0] + linear_fit_P[1] * times)

    linear_fit_Z = fit(times[Interval_3][Z_peaks.indices], log.(Z′[Interval_3][Z_peaks.indices]), degree, var = :t)
    best_fit_Z = @. exp(linear_fit_Z[0] + linear_fit_Z[1] * times)



    # ODE_PDE_system_3D(total_population, λ, ν, δ, g, m, d₁, d₂, d₃, k)[1][2]

    print("Growth rate of N is approximately ", linear_fit_N[1]," with oscilation period ", 2*mean(diff(times[Interval_1][N_peaks.indices])) , "\n")
    print("Growth rate of P is approximately ", linear_fit_P[1]," with oscilation period ", 2*mean(diff(times[Interval_2][P_peaks.indices])) , "\n")
    print("Growth rate of Z is approximately ", linear_fit_Z[1]," with oscilation period ", 2*mean(diff(times[Interval_3][Z_peaks.indices])) , "\n")
    print("Largest real part of e-value ", ODE_PDE_system_3D(total_population, λ, ν, δ, g, m, d₁, d₂, d₃, k)[1][2])

    


    plot(times, N′,label="norm(N′)", yscale = :log10, linestyle=:solid,
    lw=4, xlabel="time", ylabel="norm",title="Norm of perturbations", legend=:topleft)


    plot!(times, P′,label="norm(P′)", linestyle=:solid, lw=4)#

    plot!(times, Z′,label="norm(Z′)", linestyle=:solid, lw=4)#
    
    plot!(times[Interval_1], best_fit_N[Interval_1],label="N best fit", linestyle=:dash, lw=6)
    
    plot!(times[Interval_2], best_fit_P[Interval_2],label="P best fit", linestyle=:dash, lw=6)

    plot!(times[Interval_3], best_fit_Z[Interval_3],label="Z best fit", linestyle=:dash, lw=6)

end

```
untility functions of the 4-Species Model Here-----------------------------------

```



function equilibrium_state_4D(N_t, λ, ν, δ, α, g, m)
    ## record the length of vector N_t
    num_pt=length(N_t)
    # Pre-define the vector P_star and N_star as vector to store the equilibrium values for each N_t[i]
    P̄, N̄, Z̄, D̄=zeros(Float64,num_pt), zeros(Float64,num_pt), zeros(Float64,num_pt), zeros(Float64,num_pt)

    # running the for loop for every N_t
    for i in 1:num_pt
        
       b=1+ (ν+δ)/(g-δ) - (λ/δ)*(δ/(g-δ))^m - N_t[i] + (ν/α)*(δ)/(g-δ)
       c=  δ/(g-δ) - (λ/δ)*(δ/(g-δ))^m -N_t[i]

       poly_i = Polynomial([c, b, 1])
       ## Only consider the real roots of above polynomial for solving N
       all_roots = roots(poly_i)
       real_roots = filter(x -> isreal(x), all_roots)
       real_roots = Float64.(real(real_roots))

       ## if there is no such real roots in the restricted range [0,N_t[i]], we use -1 to denote N and P
       if isempty(filter(x -> 0 <= x <= N_t[i], real_roots))
           N̄[i] = -1
           P̄[i] = -1
           D̄[i] = -1
           Z̄[i] = -1
       else
           N̄[i] = filter(x -> 0 <= x <= N_t[i], real_roots)[1]
           P̄[i] = δ/(g-δ)
           D̄[i] = (ν/α)*(δ)/(g-δ) * N̄[i]/(1+N̄[i])
           Z̄[i] = N_t[i] - N̄[i] - P̄[i] - D̄[i]
       end
       
       ## make a double check that all N P Z are in the range of [0, N_t]
       if !(0 <= N̄[i] <= N_t[i] && 0 <= P̄[i] <= N_t[i] && 0 <= Z̄[i] <= N_t[i] && 0 <= D̄[i] <= N_t[i])
        N̄[i], P̄[i], Z̄[i] = -1, -1, -1
       end
   end
   
   return(N̄,P̄,Z̄,D̄)
end


function ODE_PDE_system_4D(Nₜ, λ, ν, δ, α, g, m, d₁, d₂, d₃, d₄, k)
    N̄,P̄,Z̄,D̄=equilibrium_state_4D(Nₜ, λ, ν, δ, α, g, m)
    result=[]
    for i in 1:length(Nₜ)
        Nᵢ,Pᵢ,Zᵢ, Dᵢ=N̄[i],P̄[i],Z̄[i],D̄[i]
        if Nᵢ == -1 && Pᵢ==-1 && Zᵢ==-1 && Dᵢ==-1
           key=[-0.1, -0.1, -0.1, Nₜ[i]]
        else
           ## Define the Jacobian matrix at equilibrium states
           A=zeros(4,4)
           A=[-ν*Pᵢ/((1+Nᵢ)^2)   -ν*(Nᵢ/(Nᵢ+1))    0      α
              ν*Pᵢ/((1+Nᵢ)^2)    ν*(Nᵢ/(Nᵢ+1))-g*Zᵢ/((1+Pᵢ)^2)-m*λ*Pᵢ^(m-1)  -g*Pᵢ/(1+Pᵢ)    0
              0                  g*Zᵢ/((1+Pᵢ)^2)    0     0
              0                  m*λ*Pᵢ^(m-1)       δ     -α]
           ## define the linear approximation matrix for PDE system
           diffusion_vec=[d₁,d₂,d₃,d₄]
           diffusion_mat=diagm(0 => -(k^2)*diffusion_vec)
           pde_mat=A+diffusion_mat
           ode_mat=A
           eigen_pde=eigen(pde_mat).values
           ## define the maximum of real part of eigenvalue
           key_eigen_pde = maximum(real(eigen_pde))
           max_index= argmax(real(eigen_pde))

           eigen_ode=eigen(ode_mat).values
           key_eigen_ode = maximum(real(eigen_ode))
           ## define the eigenvector corresponding to the largest eigenvalue
           eigen_vec=(eigen(pde_mat).vectors)[:,max_index]
           eigen_val=(eigen(pde_mat).values)[max_index]
           
           key=[key_eigen_ode, key_eigen_pde, Nₜ[i], eigen_vec, eigen_val]
           
         end
         push!(result,key)
     end
return(result)
end



function plot_pde_eigenvalues_4D(Nₜ, λ, ν, δ, α, g, m, d₁, d₂, d₃, d₄, k_range, xlim_range, ylim_range)
    eigen_pde_values = []
    
    for k in k_range
        result = ODE_PDE_system_4D(Nₜ, λ, ν, δ, α, g, m, d₁, d₂, d₃, d₄, k)[1]
        push!(eigen_pde_values, result[2])
    end

    # Plot max real part of PDE eigenvalue vs k
    p = plot(
        k_range,
        eigen_pde_values,
        xlabel = "k",
        ylabel = "Max Re(λ)",
        title = "Max Real Part of PDE Eigenvalues vs k",
        lw = 2,
        label = "Max Re(λ)",
        ylim = ylim_range,
        xlim = xlim_range
    )
    hline!([0], linestyle=:dash, color=:black, label="")
end



function Jacobian_simplication(Nₜ, λ, ν, δ, α, g, m, d₁, d₂, d₃, d₄, k)
    N̄,P̄,Z̄,D̄=equilibrium_state_4D(Nₜ, λ, ν, δ, α, g, m)
    result=[]
    for i in 1:length(Nₜ)
        Nᵢ,Pᵢ,Zᵢ, Dᵢ=N̄[i],P̄[i],Z̄[i],D̄[i]
        if Nᵢ == -1 && Pᵢ==-1 && Zᵢ==-1 && Dᵢ==-1
           key=[-0.1, -0.1, -0.1, Nₜ[i]]
        else
           ## Define the Jacobian matrix at equilibrium states
           J=zeros(4,4)
           J=[-ν*Pᵢ/((1+Nᵢ)^2)   -ν*(Nᵢ/(Nᵢ+1))    0      α
              ν*Pᵢ/((1+Nᵢ)^2)    ν*(Nᵢ/(Nᵢ+1))-g*Zᵢ/((1+Pᵢ)^2)-m*λ*Pᵢ^(m-1)  -g*Pᵢ/(1+Pᵢ)    0
              0                  g*Zᵢ/((1+Pᵢ)^2)    0     0
              0                  m*λ*Pᵢ^(m-1)       δ     -α]
           ## define the linear approximation matrix for PDE system
           A=ν*Pᵢ/((1+Nᵢ)^2)
           B=ν*(Nᵢ/(Nᵢ+1))
           C=m*λ*Pᵢ^(m-1) 
           D=g*Zᵢ/((1+Pᵢ)^2)
           E=α
           F=δ
           G=g*Pᵢ/(1+Pᵢ)

           
           key=["A: $(A)", "B: $(B)", "C: $(C)", "D: $(D)", "E: $(E)", "F: $(F)", "G: $(G)", Nₜ[i]]
           
         end
         push!(result,key)
     end
return(result)
end



function key_variable_outcome_4D(time_lim, mymodel, file_name)
    space=Array(znodes(mymodel.tracers.N))
    N_timeseries = FieldTimeSeries(file_name, "N")
    P_timeseries = FieldTimeSeries(file_name, "P")
    Z_timeseries = FieldTimeSeries(file_name, "Z")
    D_timeseries = FieldTimeSeries(file_name, "D")
    times=Array(N_timeseries.times)

    N_data = parent(N_timeseries.data[:,:,1:size(space)[1],:])
    P_data = parent(P_timeseries.data[:,:,1:size(space)[1],:])
    Z_data = parent(Z_timeseries.data[:,:,1:size(space)[1],:])
    D_data = parent(D_timeseries.data[:,:,1:size(space)[1],:])
    N_data = dropdims(N_data, dims=(1, 2))
    P_data = dropdims(P_data, dims=(1, 2))
    Z_data = dropdims(Z_data, dims=(1, 2))
    D_data = dropdims(D_data, dims=(1, 2))
    
    @assert time_lim <= times[length(times)] "time_lim can not be larger than the simulated time"
    times=times[times.<time_lim]
    time_index=size(times)[1]

    N_data=N_data[:, 1:time_index]
    P_data=P_data[:, 1:time_index]
    Z_data=Z_data[:, 1:time_index]
    D_data=D_data[:, 1:time_index]

    return N_data,P_data,Z_data,D_data,times,space
end



## Surface plot
function surface_plot_4D(N_data,P_data,Z_data,D_data,times,space)
    N_plot=surface(times, space, N_data ,xlabel="Time", ylabel="Space", title="N Surface Plot")
    P_plot=surface(times, space, P_data ,xlabel="Time", ylabel="Space", title="P Surface Plot")
    Z_plot=surface(times, space, Z_data ,xlabel="Time", ylabel="Space", title="Z Surface Plot")
    D_plot=surface(times, space, D_data ,xlabel="Time", ylabel="Space", title="D Surface Plot")

    plot(N_plot, P_plot, Z_plot, D_plot, layout=(4,1),size=(1200, 1100))
end





function heatmap_plot_4D(save_img_path, N_data, P_data, Z_data, D_data, times, space)
    idx = findall(t -> 1 ≤ t ≤ 169, times)

    # Slice data and time accordingly
    N_data_slice = N_data[:, idx]
    P_data_slice = P_data[:, idx]
    Z_data_slice = Z_data[:, idx]
    D_data_slice = D_data[:, idx]
    times_slice = times[idx]

    # Define extent
    extent = [minimum(times_slice), maximum(times_slice), minimum(space), maximum(space)]

    fig, axs = PyPlot.subplots(2, 2, figsize=(12, 12), constrained_layout=true)

    # helper function to format colorbars
    function add_colorbar(cax, ax)
        cbar = fig.colorbar(cax, ax=ax, shrink=0.8, aspect=20)
        cbar.formatter.set_scientific(false)   # disable scientific notation
        cbar.update_ticks()
    end

    # N heatmap
    cax1 = axs[1].imshow(N_data_slice, aspect="auto", cmap="inferno",
                         extent=extent, origin="lower")
    axs[1].set_title("N", fontsize=16)
    axs[1].set_xlabel("Time", fontsize=14)
    add_colorbar(cax1, axs[1])

    # P heatmap
    cax2 = axs[2].imshow(P_data_slice, aspect="auto", cmap="inferno",
                         extent=extent, origin="lower")
    axs[2].set_title("P", fontsize=16)
    axs[2].set_xlabel("Time", fontsize=14)
    axs[2].set_yticks([])
    add_colorbar(cax2, axs[2])

    # Z heatmap
    cax3 = axs[3].imshow(Z_data_slice, aspect="auto", cmap="inferno",
                         extent=extent, origin="lower")
    axs[3].set_title("Z", fontsize=16)
    axs[3].set_xlabel("Time", fontsize=14)
    add_colorbar(cax3, axs[3])

    # D heatmap
    cax4 = axs[4].imshow(D_data_slice, aspect="auto", cmap="inferno",
                         extent=extent, origin="lower")
    axs[4].set_title("D", fontsize=16)
    axs[4].set_xlabel("Time", fontsize=14)
    axs[4].set_yticks([])
    add_colorbar(cax4, axs[4])

    # Save figure
    PyPlot.savefig(save_img_path, dpi=300, bbox_inches="tight")
end

## Heatmap for power of each mode
function FFT_power_4D(save_path, N_data, P_data, Z_data, D_data, N̄, P̄, Z̄, D̄, times)
    perturbation_N = N_data .- N̄
    perturbation_P = P_data .- P̄
    perturbation_Z = Z_data .- Z̄
    perturbation_D = D_data .- D̄

    mode_values = (1:41 .- 1) / 2

    rev_N_data=reverse(perturbation_N, dims=1)
    N_data_mat=vcat(rev_N_data, perturbation_N)
    fft_coeff_N_data=zeros(size(N_data_mat))


    rev_P_data=reverse(perturbation_P, dims=1)
    P_data_mat=vcat(rev_P_data, perturbation_P)
    fft_coeff_P_data=zeros(size(P_data_mat))


    rev_Z_data=reverse(perturbation_Z, dims=1)
    Z_data_mat=vcat(rev_Z_data, perturbation_Z)
    fft_coeff_Z_data=zeros(size(Z_data_mat))

    rev_D_data=reverse(perturbation_D, dims=1)
    D_data_mat=vcat(rev_D_data, perturbation_D)
    fft_coeff_D_data=zeros(size(D_data_mat))

    for i in 1:size(P_data_mat)[2]
        fft_coeff_P_data[:,i]=abs.( fft(P_data_mat[:,i]) )
        fft_coeff_N_data[:,i]=abs.( fft(N_data_mat[:,i]) )
        fft_coeff_Z_data[:,i]=abs.( fft(Z_data_mat[:,i]) )
        fft_coeff_D_data[:,i]=abs.( fft(D_data_mat[:,i]) )
    end

    mode_range = 2:21
    mode_values = (mode_range .- 1) ./ 2

    # Use your heatmap plotting function
    heatmap_plot_4D(save_path,
        fft_coeff_N_data[mode_range, :],
        fft_coeff_P_data[mode_range, :],
        fft_coeff_Z_data[mode_range, :],
        fft_coeff_D_data[mode_range, :],
        times,
        mode_values)

end


function plot_pde_eigenvalues_4D(Nₜ, λ, ν, δ, α, g, m, d₁_values, d₂, d₃, d₄, k_range, xlim_range, ylim_range)
    p = plot(xlabel="k", ylabel="Max Re(λ)",
             title="Max Real Part of PDE Eigenvalues vs k",
             lw=2,
             xlim=xlim_range,
             ylim=ylim_range)
    
    # Loop over different d₂ values
    for d₁ in d₁_values
        eigen_pde_values = []
        for k in k_range
            result = ODE_PDE_system_4D(Nₜ, λ, ν, δ, α, g, m, d₁, d₂, d₃, d₄, k)[1]
            push!(eigen_pde_values, result[2])
        end
        # Add line to the plot for current d₂
        plot!(p, k_range, eigen_pde_values, label="d₁ = $d₁",lw=2)
    end

    # Add horizontal line at 0
    hline!(p, [0], linestyle=:dash, color=:black, label="")

    return p
end


function experiment_growth_rate_4D(file_name,total_population, I₁, I₂, I₃, I₄, λ, ν, δ, g, m, d₁, d₂, d₃, d₄, k)
    ds = NCDataset(file_name, "r")
    times = ds["time"][:]

    ## Denote the perturbation for N and P from the equilibrium state
    N′ = ds["perturbation_N"]
    P′ = ds["perturbation_P"]
    Z′ = ds["perturbation_Z"]
    D′ = ds["perturbation_D"]
    time_increment=1000/size(times)[1]

    

    Interval_1=Int(round(I₁[1]/time_increment)):Int(round(I₁[size(I₁)[1]]/time_increment))
    Interval_2=Int(round(I₂[1]/time_increment)):Int(round(I₂[size(I₂)[1]]/time_increment))
    Interval_3=Int(round(I₃[1]/time_increment)):Int(round(I₃[size(I₃)[1]]/time_increment))
    Interval_4=Int(round(I₄[1]/time_increment)):Int(round(I₄[size(I₄)[1]]/time_increment))


    target_N=N′[Interval_1]
    N_peaks=findmaxima(target_N)
    target_P=P′[Interval_2]
    P_peaks=findmaxima(target_P)
    target_Z=Z′[Interval_3]
    Z_peaks=findmaxima(target_Z)
    target_D=D′[Interval_4]
    D_peaks=findmaxima(target_D)
    
    degree = 1

    ## Fit the log of growth with line on time range I for N,P
    
    linear_fit_N = fit(times[Interval_1][N_peaks.indices], log.(N′[Interval_1][N_peaks.indices]), degree, var = :t)
    best_fit_N = @. exp(linear_fit_N[0] + linear_fit_N[1] * times)
    
    linear_fit_P = fit(times[Interval_2][P_peaks.indices], log.(P′[Interval_2][P_peaks.indices]), degree, var = :t)
    best_fit_P = @. exp(linear_fit_P[0] + linear_fit_P[1] * times)

    linear_fit_Z = fit(times[Interval_3][Z_peaks.indices], log.(Z′[Interval_3][Z_peaks.indices]), degree, var = :t)
    best_fit_Z = @. exp(linear_fit_Z[0] + linear_fit_Z[1] * times)

    linear_fit_D = fit(times[Interval_4][D_peaks.indices], log.(D′[Interval_4][D_peaks.indices]), degree, var = :t)
    best_fit_D = @. exp(linear_fit_D[0] + linear_fit_D[1] * times)




    # ODE_PDE_system_3D(total_population, λ, ν, δ, g, m, d₁, d₂, d₃, k)[1][2]

    print("Growth rate of N is approximately ", linear_fit_N[1]," with oscilation period ", 2*mean(diff(times[Interval_1][N_peaks.indices])) , "\n")
    print("Growth rate of P is approximately ", linear_fit_P[1]," with oscilation period ", 2*mean(diff(times[Interval_2][P_peaks.indices])) , "\n")
    print("Growth rate of Z is approximately ", linear_fit_Z[1]," with oscilation period ", 2*mean(diff(times[Interval_3][Z_peaks.indices])) , "\n")
    print("Growth rate of D is approximately ", linear_fit_D[1]," with oscilation period ", 2*mean(diff(times[Interval_4][D_peaks.indices])) , "\n")
    print("Largest real part of e-value ", ODE_PDE_system_4D(total_population, λ, ν, δ, α, g, m, d₁, d₂, d₃, d₄, k)[1][2])

    
    ϵ = 1e-10

    Plots.plot(times, N′,label="norm(N′)", yscale = :log10, linestyle=:solid,
    lw=4, xlabel="time", ylabel="norm",title="Norm of perturbations", legend=:bottomright)


    Plots.plot!(times, P′,label="norm(P′)", linestyle=:solid, lw=4)#

    Plots.plot!(times, Z′,label="norm(Z′)", linestyle=:solid, lw=4)#

    Plots.plot!(times, D′.+ϵ,label="norm(D′)", linestyle=:solid, lw=4)#
    
    Plots.plot!(times[Interval_1], best_fit_N[Interval_1],label="N best fit", linestyle=:dash, lw=6)
    
    Plots.plot!(times[Interval_2], best_fit_P[Interval_2],label="P best fit", linestyle=:dash, lw=6)

    Plots.plot!(times[Interval_3], best_fit_Z[Interval_3],label="Z best fit", linestyle=:dash, lw=6)

    Plots.plot!(times[Interval_4], best_fit_D[Interval_4].+ϵ,label="D best fit", linestyle=:dash, lw=6)

end


function plot_eigenvalue_heatmap_4D(save_path, Nₜ, λ, ν, δ, α, g, d₁, d₂, d₃, d₄, m_range, k_range)
    heatmap_data = zeros(length(m_range), length(k_range))

    for (i, m) in enumerate(m_range)
        for (j, k) in enumerate(k_range)
            result = ODE_PDE_system_4D(Nₜ, λ, ν, δ, α, g, m, d₁, d₂, d₃, d₄, k)[1]
            heatmap_data[i, j] = real(result[2])  # keep negative values
        end
    end

    fig, ax = subplots()

    # Make the color scale symmetric around 0
    vmax_abs = maximum(abs.(heatmap_data))

    # Display heatmap
    c = ax.imshow(
        heatmap_data,
        origin="lower",
        aspect="auto",
        extent=[minimum(k_range), maximum(k_range), minimum(m_range), maximum(m_range)],
        cmap="coolwarm",      # diverging colormap
        vmin=-vmax_abs,       # symmetric around 0
        vmax=vmax_abs
    )
    cs = ax.contour(
        k_range, m_range, heatmap_data,
        levels=[0.0],
        colors="black",
        linewidths=1.6
    )
    ax.clabel(cs, inline=true, fontsize=10, fmt="0")   # label on contour



    ax.set_xlabel("k")
    ax.set_ylabel("m")
    ax.tick_params(direction="out")
    ax.set_xticks(range(minimum(k_range), stop=maximum(k_range), length=5))
    ax.set_yticks(range(minimum(m_range), stop=maximum(m_range), length=5))

    # Add colorbar
    fig.colorbar(c, ax=ax)

    tight_layout()
    PyPlot.savefig(save_path, dpi=300)
end


function plot_combined_heatmaps_4D(k_values, Nₜ, λ, ν, δ, α, g, m, d₁_vec, d₂, d₃, d₄, ODE_PDE_system_4D; 
                                layout=(2,2), cmap="bwr", n_xticks=5, n_yticks=5,
                                output_file="combined.png")

    Nₜ = collect(Nₜ)
    d₁_vec = collect(d₁_vec)
    

    fig, axes = subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor("white")

    for (idx, k) in enumerate(k_values)
        row = div(idx - 1, 2) + 1
        col = (idx - 1) % 2 + 1
        ax = axes[row, col]

        PDE_eigen = zeros(Float64, length(d₁_vec), length(Nₜ))
        # Compute PDE eigenvalues
        for i in 1:length(d₁_vec)
            for j in 1:length(Nₜ)
                _, PDE_eigen[i,j], _ = ODE_PDE_system_4D(Nₜ, λ, ν, δ, α, g, m, d₁_vec[i], d₂, d₃, d₄, k)[j]
            end
        end

        # Diverging colormap centered at 0
        vmax = maximum(abs, PDE_eigen)
        c = ax.imshow(PDE_eigen,
                      extent=[minimum(Nₜ), maximum(Nₜ), minimum(d₁_vec), maximum(d₁_vec)],
                      aspect="auto", origin="lower",
                      cmap=cmap,
                      vmin=-vmax, vmax=vmax)  # center colormap at 0

        # Draw zero contour
        cs= ax.contour(Nₜ, d₁_vec, PDE_eigen, levels=[0.0], colors="black", linewidths=1.5)
        
        ax.clabel(cs, inline=true, fontsize=10, fmt="0")   # label on contour

        ax.set_title("k = $k")
        ax.set_xlabel(L"N_{T}")
        ax.set_ylabel(L"d_{1}")
        ax.tick_params(direction="out")
        ax.set_xticks(range(minimum(Nₜ), stop=maximum(Nₜ), length=n_xticks))
        ax.set_yticks(range(minimum(d₁_vec), stop=maximum(d₁_vec), length=n_yticks))

        fig.colorbar(c, ax=ax)
    end

    tight_layout()
    PyPlot.savefig(output_file, dpi=300, facecolor=fig.get_facecolor())
    close(fig)
end
