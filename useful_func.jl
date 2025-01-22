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

## The following function computes the equilibrium points N and P
function equilibrium_state(N_t,λ,ν)
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


function ODE_PDE_system(Nₜ,λ,ν,d₁,d₂,k)
    N̄,P̄=equilibrium_state(Nₜ,λ,ν)
    result=[]
    for i in 1:length(Nₜ)
        Nᵢ,Pᵢ=N̄[i],P̄[i]
        if Nᵢ == -1 && Pᵢ==-1
           key=[-0.1,-0.1,Nₜ[i]]
        else
           ## Define the Jacobian matrix at equilibrium states
           A=zeros(2,2)
           A=[-ν*Pᵢ/((1+Nᵢ)^2)   -1/2*λ*Pᵢ^(-1/2)
               ν*Pᵢ/((1+Nᵢ)^2)   1/2*λ*Pᵢ^(-1/2)]
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

function key_variable_outcome(mymodel,file_name)
    space=Array(znodes(mymodel.tracers.N))
    N_timeseries = FieldTimeSeries(file_name, "N")
    P_timeseries = FieldTimeSeries(file_name, "P")
    times=Array(N_timeseries.times)
    N_data = parent(N_timeseries.data[:,:,1:size(space)[1],:])
    P_data = parent(P_timeseries.data[:,:,1:size(space)[1],:])
    N_data = dropdims(N_data, dims=(1, 2))
    P_data = dropdims(P_data, dims=(1, 2))
    return N_data,P_data,times,space
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
    N_plot=surface(times, space, N_data ,xlabel="Time", ylabel="Space")
    P_plot=surface(times, space, P_data ,xlabel="Time", ylabel="Space")

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


## Calculate the Growth rate depending on our simulation
function experiment_growth_rate(file_name)
    ds = NCDataset(file_name, "r")
    times = ds["time"][:]

    ## Denote the perturbation for N and P from the equilibrium state
    N′ = ds["perturbation_N"]
    P′ = ds["perturbation_P"]
    I = 2000:4000
    
    degree = 1

    ## Fit the log of growth with line on time range I for N,P
    
    linear_fit_N = fit(times[I], log.(N′[I]), degree, var = :t)
    best_fit_N = @. exp(linear_fit_N[0] + linear_fit_N[1] * times)
    
    linear_fit_P = fit(times[I], log.(P′[I]), degree, var = :t)
    best_fit_P = @. exp(linear_fit_P[0] + linear_fit_P[1] * times)

    print("Growth rate of N is approximately ", linear_fit_N[1], "\n")
    print("Growth rate of P is approximately ", linear_fit_P[1], "\n")


    plot(times, N′,label="norm(N′)", yscale = :log10, linestyle=:solid,
    lw=4, xlabel="time", ylabel="norm",title="Norm of perturbations", legend=:topleft)

    plot!(times, P′,label="norm(P′)", linestyle=:solid, lw=4)#
    
    plot!(times, best_fit_N,label="N best fit", linestyle=:dash, lw=4)
    
    plot!(times, best_fit_P,label="P best fit", linestyle=:dash, lw=4)

end
    
    

## Heatmap for power of each mode
function FFT_power(N_data, P_data, times)
    rev_N_data=reverse(N_data, dims=1)
    N_data_mat=vcat(rev_N_data, N_data)
    fft_coeff_N_data=zeros(size(N_data_mat))

    mode_values = (1:41 .- 1) / 2

    rev_P_data=reverse(P_data, dims=1)
    P_data_mat=vcat(rev_P_data, P_data)
    fft_coeff_P_data=zeros(size(P_data_mat))

    for i in 1:size(P_data_mat)[2]
        fft_coeff_P_data[:,i]=abs.(fft(P_data_mat[:,i])).^2
        fft_coeff_N_data[:,i]=abs.(fft(N_data_mat[:,i])).^2
    end
    
    yticks_values = 2:2:40
    N_plot=heatmap(times, mode_values, fft_coeff_N_data[2:41,:], xlabel="Time", ylabel="Modes", title="Power of N",yticks=(yticks_values, string.(yticks_values)))
    P_plot=heatmap(times, mode_values, fft_coeff_P_data[2:41,:], xlabel="Time", ylabel="Modes", title="Power of P",yticks=(yticks_values, string.(yticks_values)))

    plot(N_plot, P_plot, layout=(1, 2),size=(1200, 500))

end
    