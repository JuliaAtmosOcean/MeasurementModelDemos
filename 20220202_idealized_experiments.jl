#=
 = This program is a supplement to the article "Wave-like measurement modeling
 = with consensus solutions".  It provides an experimental errors-in-variables
 = solution using two idealized and imperfect measures, one taken as a reference
 = for the other arbitrarily (both share the same baseline signal).  Execution
 = may be by something like

   julia 20220202_idealized_experiments.jl 20 1 0410 a 0200 a

 = where the first two numbers are usually fixed, the third may be one of ["0410",
 = "0420", "0430", "0440"], the fourth one of ["a", "b", "c", "d", "e", "f", "g",
 = "h", "i", "j"], the fifth one of ["0200", "0400", "0600", "0800", "1000"], and
 = the sixth one of ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"].  Numbers
 = of the third and fifth arguments refer to Spearman and Gaussian weight; letters
 = of the fourth and sixth arguments refer to a particular set of perturbations.
 = Other terms are defined in the article.  Only the following standard packages
 = (see "using" below, with julia version 1.4) should be needed - RD October 2020,
 = August 2021.
 =#

using Printf, FFTW, Random, NetCDF, Statistics, StatsBase

const RRCM             = 10                             # row    dimension of RCM (ABCDE/STUVW samples)
const CRCM             = 19                             # column dimension of RCM (var/cov/cal/val metrics)
const MRCM             = 3                              # metric dimension of RCM (MOLR/MEIV/MRLR solutions)
const RRAA             = 1                              # extended forecast   calibrated
const RRBB             = 2                              #          forecast   calibrated
const RRCC             = 3                              #           nowcast   calibrated
const RRDD             = 4                              #           revcast   calibrated
const RREE             = 5                              # extended  revcast   calibrated
const RRSS             = 6                              # extended forecast uncalibrated
const RRTT             = 7                              #          forecast uncalibrated
const RRUU             = 8                              #           nowcast uncalibrated
const RRVV             = 9                              #           revcast uncalibrated
const RRWW             = 10                             # extended  revcast uncalibrated
const PNUM             = 11                             # number of nonoutliers                              (part of avg but not rcm)
const PREA             = 12                             # precalibration alpha                               (part of avg but not rcm)
const PREB             = 13                             # precalibration beta                                (part of avg but not rcm)
const PTTT             = 14                             # target variance in linear association/shared truth (part of avg but not rcm)
const PTBB             = 15                             # target multiplicative calibration                  (part of avg but not rcm)
const PFTT             = 16                             # final  variance in linear association/shared truth (part of avg but not rcm)
const PFBB             = 17                             # final  multiplicative calibration                  (part of avg but not rcm)
const PDIS             = 18                             # normalized distance between target and final EIV solution  (avg but not rcm)
const CCVA             = 1                              # covariance with A                           (ABCDE STUVW)
const CCVB             = 2                              # covariance with B                            (BCDE STUVW)
const CCVC             = 3                              # covariance with C                             (CDE STUVW)
const CCVD             = 4                              # covariance with D                              (DE STUVW)
const CCVE             = 5                              # covariance with E                               (E STUVW)
const CCVS             = 6                              # covariance with S                                 (STUVW)
const CCVT             = 7                              # covariance with T                                  (TUVW)
const CCVU             = 8                              # covariance with U                                   (UVW)
const CCVV             = 9                              # covariance with V                                    (VW)
const CCVW             = 10                             # covariance with W                                     (W)
const CTRU             = 11                             # variance in linear association/shared truth (ABCDE=STUVW)
const CALP             = 12                             # additive       calibration                  (AB DE STUVW)
const CBET             = 13                             # multiplicative calibration                  (AB DE STUVW)
const CLAM             = 14                             # autoregressive shared error fraction        (AB DE ST VW)
const CERI             = 15                             # error variance individual                   (ABCDE STUVW)
const CERT             = 16                             # error variance combined                     (ABCDE STUVW)
const CLIN             = 17                             # percent variance in    linear association     (C     U)
const CNOL             = 18                             # percent variance in nonlinear association     (C     U)
const CNOA             = 19                             # percent variance in a lack of association     (C     U)
const MOLR             = 1                              # ordinary linear regression metrics
const MEIV             = 2                              # causal-predictive-sampling metrics
const MRLR             = 3                              # reverse  linear regression metrics

const SEPBASE          = true                           # use a separate timeseries for each baseline sample
const PLOTPROG         = false                          # required plotting program (GrADs) is available
const PLOTCOST         = false                          # include plots of the location of the minimum cost
const KEEPNETCDF       = false                          # retain all data files in order to reproduce figures
const DETMCDUSE        = false                          # Minimum Covariance Determinant use (i.e., with trimming of outliers)
const TRIMMCD          = 0.95                           # Minimum Covariance Determinant trimming (nonoutlier percent is higher)
const FULLBIN          = -0.70:0.010:0.70               # binning for display of     full timeseries variations
const BANDBIN          = -0.25:0.003:0.25               # binning for display of bandpass timeseries variations
const DELT             = 1e-9                           # small number for comparison
const MISS             = -9999.0                        # generic missing value
const BUTORD           = 5.0                            # Butterworth filter order

if (argc = length(ARGS)) != 6
  print("\nUsage: jjj $(basename(@__FILE__)) texp sampint 0020 h 0000 a\n")
  print("       where texp is the power-of-two exponent of timeseries length\n")
  print("       and sampint is the interval between predictive samples\n\n")
  exit(1)
end
texp = parse(Int64, ARGS[1])
sint = parse(Int64, ARGS[2])
tsst = @sprintf("%4d.%4d", texp, sint) ; tsst = replace(tsst, ' ' => '0')

ntim = 2^texp                                                                 # first set all values of time and frequency
tims = collect(range(1900, step = 1 / 24 / 365, length = ntim))
frqs = rfftfreq(ntim)[:]
nfrq = length(frqs)
@printf("  timeseries length is %9d\n", ntim)
@printf("   frequency length is %9d\n", nfrq)

fila = "spur.$tsst.0000.time.nc"                                              # then create the data files, as needed
filg = "spur.$tsst.0000.cols.nc"
filh = "spur.$tsst.0000.rots.nc"
fili = "spur.$tsst.0000.epss.nc"
filj = "spur.$tsst.0000.base.nc"

function nccreer(fn::AbstractString, ntim::Int, nlat::Int, nlon::Int, missing::Float64; vnames = ["tmp"])
  nctim = NcDim("time", ntim, atts = Dict{Any,Any}("units"=>"hours since 1-1-1 00:00:0.0"), values = collect(range(    0, stop =                  ntim - 1 , length = ntim)))
  nclat = NcDim( "lat", nlat, atts = Dict{Any,Any}("units"=>              "degrees_north"), values = collect(range( 50.0, stop =  50.0 + 0.001 * (nlat - 1), length = nlat)))
  nclon = NcDim( "lon", nlon, atts = Dict{Any,Any}("units"=>               "degrees_east"), values = collect(range(280.0, stop = 280.0 + 0.001 * (nlon - 1), length = nlon)))
  ncvrs = Array{NetCDF.NcVar}(undef, length(vnames))
  for a = 1:length(vnames)
    ncvrs[a] = NcVar(vnames[a], [nclon, nclat, nctim], atts = Dict{Any,Any}("units"=>"none", "missing_value"=>missing), t=Float64, compress=-1)
  end
  ncfil = NetCDF.create(fn, ncvrs, gatts = Dict{Any,Any}("units"=>"none"), mode = NC_NETCDF4)
  print("created $fn with $ntim times $nlat lats and $nlon lons\n")
  return
end

function nccreer(fn::AbstractString, ntim::Int, nlev::Int, nlat::Int, nlon::Int, missing::Float64; vnames = ["tmp"])
  nctim = NcDim(  "time", ntim, atts = Dict{Any,Any}("units"=>"hours since 1-1-1 00:00:0.0"), values = collect(range(    0, stop =                  ntim - 1 , length = ntim)))
  nclev = NcDim( "level", nlev, atts = Dict{Any,Any}("units"=>                      "level"), values = collect(range(    1, stop =                  nlev     , length = nlev)))
  nclat = NcDim(   "lat", nlat, atts = Dict{Any,Any}("units"=>              "degrees_north"), values = collect(range( 50.0, stop =  50.0 + 0.001 * (nlat - 1), length = nlat)))
  nclon = NcDim(   "lon", nlon, atts = Dict{Any,Any}("units"=>               "degrees_east"), values = collect(range(280.0, stop = 280.0 + 0.001 * (nlon - 1), length = nlon)))
  ncvrs = Array{NetCDF.NcVar}(undef, length(vnames))
  for a = 1:length(vnames)
    ncvrs[a] = NcVar(vnames[a], [nclon, nclat, nclev, nctim], atts = Dict{Any,Any}("units"=>"none", "missing_value"=>missing), t=Float64, compress=-1)
  end
  ncfil = NetCDF.create(fn, ncvrs, gatts = Dict{Any,Any}("units"=>"none"), mode = NC_NETCDF4)
  print("created $fn with $ntim times $nlev levels $nlat lats and $nlon lons\n")
  return
end

function nccreer(fn::AbstractString, ntim::Int, lats::Array{T,1}, lons::Array{T,1}, missing::Float64; vnames = ["tmp"]) where {T<:Real}
  nctim = NcDim("time",         ntim, atts = Dict{Any,Any}("units"=>"hours since 1-1-1 00:00:0.0"), values = collect(range(0, stop = ntim - 1, length = ntim)))
  nclat = NcDim( "lat", length(lats), atts = Dict{Any,Any}("units"=>              "degrees_north"), values = lats)
  nclon = NcDim( "lon", length(lons), atts = Dict{Any,Any}("units"=>               "degrees_east"), values = lons)
  ncvrs = Array{NetCDF.NcVar}(undef, length(vnames))
  for a = 1:length(vnames)
    ncvrs[a] = NcVar(vnames[a], [nclon, nclat, nctim], atts = Dict{Any,Any}("units"=>"none", "missing_value"=>missing), t=Float64, compress=-1)
  end
  ncfil = NetCDF.create(fn, ncvrs, gatts = Dict{Any,Any}("units"=>"none"), mode = NC_NETCDF4)
  VERSION < v"1" && NetCDF.close(ncfil)
  print("created $fn with $ntim times $(length(lats)) lats and $(length(lons)) lons\n")
  return
end

if !isfile(fila)
  temp = rand(ntim)                                                           # create a measureable truth timeseries consisting
  ttal = temp .- mean(temp)                                                   # of random and uniform samples on [-0.5, 0.5] and
  tsal = rfft(ttal)                                                           # get the Fourier transform of this timeseries;
  temp = randn(ntim) ./ 12^0.5                                                # similarly create two error timeseries consisting
  etal = temp .- mean(temp)                                                   # of Gaussian samples (with power/variance divided
  esal = rfft(etal)                                                           # by 12 to be equivalent to the true timeseries)
  temp = randn(ntim) ./ 12^0.5
  ftal = temp .- mean(temp)
  fsal = rfft(ftal)

  cutlow = 1 / 8760                                                           # define high and low frequency cutoffs at a day
  cutmed = 1 / 24                                                             # and a year (taking sampled data to be hourly)
  tslo = deepcopy(tsal)                                                       # and filter to remove these low and high freq
  tsme = deepcopy(tsal)                                                       # variations (we focus on the midrange below)
  tshi = deepcopy(tsal)
  eslo = deepcopy(esal)
  esme = deepcopy(esal)
  eshi = deepcopy(esal)
  for a = 1:nfrq
    filtlow = 1 / (1 + (frqs[a] / cutlow)^(2 * BUTORD))
    filtmed = 1 / (1 + (frqs[a] / cutmed)^(2 * BUTORD))
    tslo[a] *= filtlow
    tsme[a] *= filtmed * (1.0 - filtlow)
    tshi[a] -= tslo[a] + tsme[a]
    eslo[a] *= filtlow
    esme[a] *= filtmed * (1.0 - filtlow)
    eshi[a] -= eslo[a] + esme[a]
  end
  ttlo = irfft(tslo, ntim)
  ttme = irfft(tsme, ntim)
  tthi = ttal .- ttlo .- ttme
  etlo = irfft(eslo, ntim)
  etme = irfft(esme, ntim)
  ethi = etal .- etlo .- etme

  vars = ["ttal", "ttlo", "ttme", "tthi", "etal", "etlo", "etme", "ethi", "ftal", "ttsb"]
  nccreer(       fila, 1, ntim, 1, 1, MISS; vnames = vars)
  ncwrite( ttal, fila,  "ttal", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( ttlo, fila,  "ttlo", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( ttme, fila,  "ttme", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( tthi, fila,  "tthi", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( etal, fila,  "etal", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( etlo, fila,  "etlo", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( etme, fila,  "etme", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( ethi, fila,  "ethi", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( ftal, fila,  "ftal", start=[1,1,1,1], count=[-1,-1,-1,-1])
  ncwrite( tims, fila, "level", start=[1],       count=[-1])
  ncwrite([0.0], fila,  "time", start=[1],       count=[-1])
  ncputatt(      fila,  "time", Dict("units" => "hours since 2020-01-01 00:00:0.0"))

  if SEPBASE                                                                  # include a timeseries that is a bunch
    tind = 51 ; while tind <= 1009950                                         # of 101-segments from the middle of 
      temp = rand(ntim)                                                       # separate baselines, strung together
      ttal = temp .- mean(temp)
      tsal = rfft(ttal)
      for a = 1:nfrq
        filtlow = 1 / (1 + (frqs[a] / cutlow)^(2 * BUTORD))
        filtmed = 1 / (1 + (frqs[a] / cutmed)^(2 * BUTORD))
        tsal[a] *= filtmed * (1.0 - filtlow)
      end
      temp = irfft(tsal, ntim) ;# print("to $(tind+50)\n")
      ttme[tind-50:tind+50] = temp[div(ntim,2)-50:div(ntim,2)+50]
      global tind += 101
    end
    ncwrite(ttme,           fila,  "ttsb", start=[1,1,1,1], count=[-1,-1,-1,-1])
  end
end

if !isfile(filg)
  cols = Array{Int64}(undef, 0)                                               # define samples of the timeseries
  if SEPBASE                                                                  # at intervals of about 100 (either
    tind = 51 ; while tind <= ntim - 50 && length(cols) < 10000               # exactly 101 or randomly spaced)
      push!(cols, tind)
      global tind += 101
    end
  else
    tind = 50 ; while tind < ntim - 50
      push!(cols, tind)
      global tind += rand(10:190)
    end
  end
  ncol = length(cols)

  vars = ["cols"]
  nccreer(filg, 1, ncol, 1, 1, MISS; vnames = vars)
  ncwrite(cols, filg, "cols", start=[1,1,1,1], count=[-1,-1,-1,-1])
end

cols = convert.(Int64, ncread(filg, "cols", start=[1,1,1,1], count=[-1,-1,-1,-1])[:])
ncol = length(cols)
@printf(" collocation length is %9d\n", ncol)

if !isfile(filh)                                                              # save Spearman perturbations
  vars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
  nccreer(filh, ncol, 1, 1, MISS; vnames = vars)
  for namr in vars
    temp = randn(ncol)
    ncwrite(temp, filh, namr, start=[1,1,1], count=[-1,-1,-1])
  end
end

if !isfile(fili)                                                              # save Gaussian perturbations
  vars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
  nccreer(fili, ntim, 1, 1, MISS; vnames = vars)
  for name in vars
    temp = randn(ntim)
    eeee = temp .- mean(temp)
    ncwrite(eeee, fili, name, start=[1,1,1], count=[-1,-1,-1])
  end
end

if !isfile(filj)                                                              # save ancillary perturbations
  vars = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
  nccreer(filj, ntim, 1, 1, MISS; vnames = vars)
  for name in vars
    temp = randn(ntim)
    eeee = temp .- mean(temp)
    ncwrite(eeee, filj, name, start=[1,1,1], count=[-1,-1,-1])
  end
end

#=
 = This section provides solutions of an errors-in-variables (EIV) regression model
 = that linearly relates uncalibrated (U) and calibrated (C) data, where calibration is
 = partial and what it means to be calibrated is left to the user (caveat emptor).  Model
 = solutions employ a causal (instrumental-variable) method called predictive sampling,
 = which extends the partial linear relationship between U and C to samples that are
 = nearly collocated in space or time with either U, C, or both.  Specifically, solutions
 = depend on our provision of symmetric samples (both before/left and after/right of U
 = and/or C) and are provided for one bracketing pair of samples in both U and C, and for
 = two bracketing pairs in either U or C (no bracketing pairs in C or U, respectively).
 = Each solution includes three sets of calibration and performance metrics, with two
 = being the bounding ordinary (MOLR) and reverse (MRLR) linear regression metrics and
 = the third being the relevant predictive sampling (MEIV) metrics.
 =#

const ESTIM            = 500                            # number of true variance and calibration slope estimates
const MCDTRIM          = 0.90                           # Minimum Covariance Determinant trimming (nonoutlier fraction)
const CUTOFF           = 5.0                            # arbitrary cutoff of log(abs(variance)) for display only
const D2R              = 3.141592654 / 180.0            # degrees to radians conversion

function logfithist(vec::Array{Float64,1},                        rng::StepRangeLen{Float64})  map(x -> x == 0 ? -1.0 : log10(x), fit(Histogram,  vec,             rng ; closed = :left).weights)  end

function logfitdoub(vec::Array{Float64,1}, ved::Array{Float64,1}, rng::StepRangeLen{Float64})  map(x -> x == 0 ? -1.0 : log10(x), fit(Histogram, (vec, ved), (rng, rng); closed = :left).weights)  end

function statline(vec::Array{Float64,1})  @sprintf("%7.3f %7.3f %7.3f %7.3f\n", mean(vec), std(vec), skewness(vec), 3.0 + kurtosis(vec))  end

function ecdfsup(x::Array{Float64,1}, y::Array{Float64,1})
  nx, ny = length(x), length(y) ; sort_idx = sortperm([x; y])                 # compute supremum of differences between empirical cdfs
  pdf_diffs = [ones(nx)/nx; -ones(ny)/ny][sort_idx]                           # (from last function of https://github.com/JuliaStats/
  cdf_diffs = cumsum(pdf_diffs)                                               # HypothesisTests.jl/blob/master/src/kolmogorov_smirnov.jl
  max(maximum(cdf_diffs), -minimum(cdf_diffs))                                # and compute the sum of histogram differences (difhist)
end

function difhist(x::Array{Float64,1}, y::Array{Float64,1}, rng::StepRangeLen{Float64})
  sum(abs.(fit(Histogram, x, rng ; closed = :left).weights .- fit(Histogram, y, rng ; closed = :left).weights)) / 2.0
end

function coarsen(mask::BitArray{2}, mtmp::BitArray{2}, extn::Int64)
  for a = 1:ESTIM, b = 1:ESTIM  mtmp[a,b] = mask[a,b]  end
  for a = 1:ESTIM, b = 1:ESTIM
    if mask[a,b]
      for c = -extn:extn, d = -extn:extn  1 <= a+c <= ESTIM && 1 <= b+d <= ESTIM && (mtmp[a+c,b+d] = true)  end
    end
  end
  for a = 1:ESTIM, b = 1:ESTIM  mask[a,b] = mtmp[a,b]  end
end

function coarsen(grid::Array{Float64,2}, step::Int64, extn::Int64)
  copy = zeros(ESTIM, ESTIM)
  for z = 1:step
    for a = 1:ESTIM, b = 1:ESTIM
      if grid[a,b] == z
        for c = -extn:extn, d = -extn:extn  1 <= a+c <= ESTIM && 1 <= b+d <= ESTIM && (copy[a+c,b+d] = z)  end
      end
    end
  end
  for a = 1:ESTIM, b = 1:ESTIM  grid[a,b] = copy[a,b]  end
end

function consensus(solve::Bool, rngtt::Array{Float64,1}, rngbu::Array{Float64,1}, est00::Array{Float64,2}, est01::Array{Float64,2}, est02::Array{Float64,2}, est03::Array{Float64,2}, est04::Array{Float64,2}, est05::Array{Float64,2}, est06::Array{Float64,2})
  esttt =  zeros(ESTIM, ESTIM)
  estbu =  zeros(ESTIM, ESTIM)
  msk01 = falses(ESTIM, ESTIM)
  msk02 = falses(ESTIM, ESTIM)
  msk03 = falses(ESTIM, ESTIM)
  msk04 = falses(ESTIM, ESTIM)
  msk05 = falses(ESTIM, ESTIM)
  msk06 = falses(ESTIM, ESTIM)

  for loop in 1:6
    loop == 1 && (grid = est01 ; mask = msk01)
    loop == 2 && (grid = est02 ; mask = msk02)
    loop == 3 && (grid = est03 ; mask = msk03)
    loop == 4 && (grid = est04 ; mask = msk04)
    loop == 5 && (grid = est05 ; mask = msk05)
    loop == 6 && (grid = est06 ; mask = msk06)
    mskt = falses(ESTIM, ESTIM)
    mskb = falses(ESTIM, ESTIM)
    mtmp = falses(ESTIM, ESTIM)
    gdir =  zeros(ESTIM, ESTIM)
    gtmp =  zeros(ESTIM, ESTIM)

    for a = 1:ESTIM, b = 1:ESTIM  gtmp[a,b] = grid[a,b]  end                  # first smooth each input grid
    for a = 2:ESTIM-1, b = 2:ESTIM-1
      sum = 0.0
      for c = -1:1, d = -1:1  sum += grid[a+c,b+d]  end
      gtmp[a,b] = sum / 9
    end
    for a = 1:ESTIM, b = 1:ESTIM  grid[a,b] = gtmp[a,b]  end

    for a = 2:ESTIM-1, b = 2:ESTIM-1                                          # and using a smoothed gradient
      gradb  = (3 * grid[a-1,b+1] -  3 * grid[a-1,b-1] +                      # define an along-gradient grid
               10 * grid[a  ,b+1] - 10 * grid[a  ,b-1] +                      # connectivity
                3 * grid[a+1,b+1] -  3 * grid[a+1,b-1]) / 32
      gradt  = (3 * grid[a+1,b-1] + 10 * grid[a+1,b  ] + 3 * grid[a+1,b+1] -
                3 * grid[a-1,b-1] - 10 * grid[a-1,b  ] - 3 * grid[a-1,b+1]) / 32
      gradh =     gradt^2 - gradb^2
      gradv = 2 * gradt   * gradb
      sdir  = atan(gradv, gradh) / D2R
          if -135 < sdir <=  -45  gdir[a,b] = 2                               # define NE-SW connections
      elseif  -45 < sdir <=   45  gdir[a,b] = 3                               #         N-S
      elseif   45 < sdir <=  135  gdir[a,b] = 4                               #        NW-SE
      else                        gdir[a,b] = 1                               #         E-W
      end
    end

    for a = 2:ESTIM-1
      gmin = findmin(grid[a,:])[2] ; 1 < gmin < ESTIM && (mskt[a,gmin] = true)
    end
    for b = 2:ESTIM-1
      gmin = findmin(grid[:,b])[2] ; 1 < gmin < ESTIM && (mskb[gmin,b] = true)
    end
    for a = 2:ESTIM-1, b = 2:ESTIM-1                                          # on rngtt and rngbu slices, get
      mskt[a,b] && mskb[a,b] && (mask[a,b] = true)                            # separate minima (mskt and mskb)
    end                                                                       # and their combination (mask)
    coarsen(mskt, mtmp, 3)                                                    # and coarsen each to facilitate
    coarsen(mskb, mtmp, 3)                                                    # connections along the gradient
    coarsen(mask, mtmp, 2)

    extn = 1                                                                  # then extend the combined minima
    while extn > 0                                                            # where the separate slice minima
      extn = 0                                                                # exist (only along the gradient)
      for a = 1:ESTIM,   b = 1:ESTIM  mtmp[a,b] = mask[a,b]  end
      for a = 2:ESTIM-1, b = 2:ESTIM-1
        if mask[a,b]
          if     gdir[a,b] == 1
            (mskt[a-1,b  ] || mskb[a-1,b  ]) && mtmp[a-1,b  ] == false && (mtmp[a-1,b  ] = true ; extn += 1)
            (mskt[a+1,b  ] || mskb[a+1,b  ]) && mtmp[a+1,b  ] == false && (mtmp[a+1,b  ] = true ; extn += 1)
          elseif gdir[a,b] == 2
            (mskt[a-1,b-1] || mskb[a-1,b-1]) && mtmp[a-1,b-1] == false && (mtmp[a-1,b-1] = true ; extn += 1)
            (mskt[a+1,b+1] || mskb[a+1,b+1]) && mtmp[a+1,b+1] == false && (mtmp[a+1,b+1] = true ; extn += 1)
          elseif gdir[a,b] == 3
            (mskt[a  ,b-1] || mskb[a  ,b-1]) && mtmp[a  ,b-1] == false && (mtmp[a  ,b-1] = true ; extn += 1)
            (mskt[a  ,b+1] || mskb[a  ,b+1]) && mtmp[a  ,b+1] == false && (mtmp[a  ,b+1] = true ; extn += 1)
          elseif gdir[a,b] == 4
            (mskt[a+1,b-1] || mskb[a+1,b-1]) && mtmp[a+1,b-1] == false && (mtmp[a+1,b-1] = true ; extn += 1)
            (mskt[a-1,b+1] || mskb[a-1,b+1]) && mtmp[a-1,b+1] == false && (mtmp[a-1,b+1] = true ; extn += 1)
          end
        end
      end
#     print("added $extn gridboxes to mask\n")
      for a = 1:ESTIM, b = 1:ESTIM  mask[a,b] = mtmp[a,b]  end
    end
    coarsen(mask, mtmp, 1)
  end

  rngin = collect(1:ESTIM)
  for a = 2:ESTIM-1                                                           # get consensus minima on rngtt slices
    estttmin = Array{Float64}(undef, 0)
    est01min = mean(rngin[msk01[a,:]]) ; !isnan(est01min) && push!(estttmin, est01min)
    est02min = mean(rngin[msk02[a,:]]) ; !isnan(est02min) && push!(estttmin, est02min)
    est03min = mean(rngin[msk03[a,:]]) ; !isnan(est03min) && push!(estttmin, est03min)
    est04min = mean(rngin[msk04[a,:]]) ; !isnan(est04min) && push!(estttmin, est04min)
    est05min = mean(rngin[msk05[a,:]]) ; !isnan(est05min) && push!(estttmin, est05min)
    est06min = mean(rngin[msk06[a,:]]) ; !isnan(est06min) && push!(estttmin, est06min)
    if length(estttmin) != 0
      coind = round(Int, mean(estttmin))
      esttt[a,coind] = length(estttmin)
    end
  end
  for b = 2:ESTIM-1                                                           # get consensus minima on rngbu slices
    estbumin = Array{Float64}(undef, 0)
    est01min = mean(rngin[msk01[:,b]]) ; !isnan(est01min) && push!(estbumin, est01min)
    est02min = mean(rngin[msk02[:,b]]) ; !isnan(est02min) && push!(estbumin, est02min)
    est03min = mean(rngin[msk03[:,b]]) ; !isnan(est03min) && push!(estbumin, est03min)
    est04min = mean(rngin[msk04[:,b]]) ; !isnan(est04min) && push!(estbumin, est04min)
    est05min = mean(rngin[msk05[:,b]]) ; !isnan(est05min) && push!(estbumin, est05min)
    est06min = mean(rngin[msk06[:,b]]) ; !isnan(est06min) && push!(estbumin, est06min)
    if length(estbumin) != 0
      coind = round(Int, mean(estbumin))
      estbu[coind,b] = length(estbumin)
    end
  end

  smoopass = 0                                                                # find the maximum number of consensus
  estmp = esttt + estbu                                                       # minima on both rngtt and rngbu slices;
  estmpmax, estmppos = findmax(estmp)                                         # if this is not unique then successively
  estmpmsk = zeros(ESTIM, ESTIM) ; estmpmsk[estmp .== estmpmax] .= 1.0        # pass a nine-point smoother until it is
  estmplen = length(findall(x -> x == estmpmax, estmp))                       # (this becomes the target minimum), but
  print("\nno smoothing yields $estmplen maxima\n")                           # constrain the search to non-unique set
  if estmppos == 1
    estmplen =  1
    smoopass = -1
    solve && print("\ncpseiv ERROR : no target solution is available\n\n")
    solve = false
  end
  while estmplen > 1
    smtmp = zeros(ESTIM, ESTIM)
    for a = 2:length(rngtt)-1, b = 2:length(rngbu)-1
      smtmp[a,b] = (estmp[a-1,b-1] + estmp[a-1,b] + estmp[a-1,b+1] + estmp[a,b-1] + estmp[a,b] + estmp[a,b+1] + estmp[a+1,b-1] + estmp[a+1,b] + estmp[a+1,b+1]) / 9
    end
    smoopass += 1
    estmp = smtmp
    estmpmax, estmppos = findmax(estmp .* estmpmsk)
    estmplen = length(findall(x -> x == estmpmax, estmp .* estmpmsk))
    print("$smoopass pass of a nine-point smoother yields $estmplen maxima\n")
  end

  finit, finib = tarit, tarib = Tuple(estmppos)                               # then find the nearest unmasked tttt and
  if solve && est00[finit,finib] != 0                                         # betu (this becomes the final minimum)
    local mindis = 9e99
    for a = 2:length(rngtt)-1, b = 2:length(rngbu)-1
      if est00[a,b] == 0
        tmpdis = (a - tarit)^2 + (b - tarib)^2
        mindis > tmpdis && ((mindis, finit, finib) = (tmpdis, a, b))
      end
    end
  end

  coarsen(esttt, 6, 3)                                                        # and return coarse grids for visualization
  coarsen(estbu, 6, 3)
  return(solve, tarit, tarib, finit, finib, esttt, estbu, smoopass, msk01, msk02, msk03, msk04, msk05, msk06)
end

function cpseiv(cc::Array{Float64,1}, precalalp::Float64, precalbet::Float64, ss::Array{Float64,1}, tt::Array{Float64,1}, uu::Array{Float64,1}, vv::Array{Float64,1}, ww::Array{Float64,1}; missval = MISS, detmcd = true, limmcd = MCDTRIM, pic = "", picrng = 1.0:0.0, keepnc = false, echotxt = [])
  avg    = fill(missval, PDIS)
  rcm    = fill(missval, (RRCM, CRCM, MRCM))
  sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", missval, missval, missval, missval, missval, missval)
  length(cc) < 50 && return(avg, rcm, sumlin)

# if detmcd                                                                   # get a mask to exclude outliers using DetMCD in R
#   temp = [cc ss tt uu vv ww]                                                # (before precalibration, avgc = remp[:center][1],
#   remp = rcopy(R"DetMCD($temp, alpha = $limmcd)")                           # varc = remp[:cov][1,1], cvcu = remp[:cov][1,4])
#   mask = falses(length(cc)) ; for a in remp[:Hsubsets]  mask[a] = true  end
# else
    mask =  trues(length(cc))
# end

  if precalalp == MISS && precalbet == MISS
    avgc = mean(cc[mask]) ; varc = var(cc[mask])
    avgu = mean(uu[mask]) ; varu = var(uu[mask])
    cvcu = cov(cc[mask], uu[mask])
    precalbet = sign(cvcu) * (varu / varc)^0.5
    precalalp = avgu - precalbet * avgc
    @printf("\npreliminary calibration by variance matchng alp = %15.8f bet = %15.8f\n", precalalp, precalbet)
  else
    @printf("\npreliminary calibration by specified values alp = %15.8f bet = %15.8f\n", precalalp, precalbet)
  end
  for a = 1:length(uu)
    ss[a] -= precalalp ; ss[a] /= precalbet
    tt[a] -= precalalp ; tt[a] /= precalbet
    uu[a] -= precalalp ; uu[a] /= precalbet
    vv[a] -= precalalp ; vv[a] /= precalbet
    ww[a] -= precalalp ; ww[a] /= precalbet
  end

  avgc = mean(cc[mask]) ; varc = var(cc[mask])
  avgs = mean(ss[mask]) ; vars = var(ss[mask])
  avgt = mean(tt[mask]) ; vart = var(tt[mask])
  avgu = mean(uu[mask]) ; varu = var(uu[mask])
  avgv = mean(vv[mask]) ; varv = var(vv[mask])
  avgw = mean(ww[mask]) ; varw = var(ww[mask])
  cvcs = cov(cc[mask], ss[mask])
  cvct = cov(cc[mask], tt[mask])
  cvcu = cov(cc[mask], uu[mask])
  cvcv = cov(cc[mask], vv[mask])
  cvcw = cov(cc[mask], ww[mask])
  cvst = cov(ss[mask], tt[mask])
  cvsu = cov(ss[mask], uu[mask])
  cvsv = cov(ss[mask], vv[mask])
  cvsw = cov(ss[mask], ww[mask])
  cvtu = cov(tt[mask], uu[mask])
  cvtv = cov(tt[mask], vv[mask])
  cvtw = cov(tt[mask], ww[mask])                                              # get RCM OLR (no C error) and RLR (no U error) values
  cvuv = cov(uu[mask], vv[mask])                                              # for available predictive samples among ABCDE and STUVW
  cvuw = cov(uu[mask], ww[mask])                                              # (for simplicity, CLAM = 0 for OLR-STVW and RLR-ABDE)
  cvvw = cov(vv[mask], ww[mask])
  avg = [MISS, MISS, avgc, MISS, MISS, avgs, avgt, avgu, avgv, avgw, length(cc[mask]), precalalp, precalbet, MISS, MISS, MISS, MISS, MISS]

  rcm = zeros(RRCM, CRCM, MRCM)
  a = RRCC ; rcm[a,CCVC,:] .= varc ; rcm[a,CCVS,:] .= cvcs ; rcm[a,CCVT,:] .= cvct ; rcm[a,CCVU,:] .= cvcu ; rcm[a,CCVV,:] .= cvcv ; rcm[a,CCVW,:] .= cvcw
  a = RRSS ; rcm[a,CCVC,:] .= cvcs ; rcm[a,CCVS,:] .= vars ; rcm[a,CCVT,:] .= cvst ; rcm[a,CCVU,:] .= cvsu ; rcm[a,CCVV,:] .= cvsv ; rcm[a,CCVW,:] .= cvsw
  a = RRTT ; rcm[a,CCVC,:] .= cvct ; rcm[a,CCVS,:] .= cvst ; rcm[a,CCVT,:] .= vart ; rcm[a,CCVU,:] .= cvtu ; rcm[a,CCVV,:] .= cvtv ; rcm[a,CCVW,:] .= cvtw
  a = RRUU ; rcm[a,CCVC,:] .= cvcu ; rcm[a,CCVS,:] .= cvsu ; rcm[a,CCVT,:] .= cvtu ; rcm[a,CCVU,:] .= varu ; rcm[a,CCVV,:] .= cvuv ; rcm[a,CCVW,:] .= cvuw
  a = RRVV ; rcm[a,CCVC,:] .= cvcv ; rcm[a,CCVS,:] .= cvsv ; rcm[a,CCVT,:] .= cvtv ; rcm[a,CCVU,:] .= cvuv ; rcm[a,CCVV,:] .= varv ; rcm[a,CCVW,:] .= cvvw
  a = RRWW ; rcm[a,CCVC,:] .= cvcw ; rcm[a,CCVS,:] .= cvsw ; rcm[a,CCVT,:] .= cvtw ; rcm[a,CCVU,:] .= cvuw ; rcm[a,CCVV,:] .= cvvw ; rcm[a,CCVW,:] .= varw
  c = MOLR
  a = RRCC ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] =         1.0 ; rcm[a,CALP,c] =                         0.0
  a = RRSS ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcs / varc ; rcm[a,CALP,c] = avgs - rcm[a,CBET,c] * avgc
  a = RRTT ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvct / varc ; rcm[a,CALP,c] = avgt - rcm[a,CBET,c] * avgc
  a = RRUU ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcu / varc ; rcm[a,CALP,c] = avgu - rcm[a,CBET,c] * avgc
  a = RRVV ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcv / varc ; rcm[a,CALP,c] = avgv - rcm[a,CBET,c] * avgc
  a = RRWW ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcw / varc ; rcm[a,CALP,c] = avgw - rcm[a,CBET,c] * avgc
  a = RRCC ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] =                         0.0
  a = RRSS ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vars - rcm[a,CBET,c] * cvcs
  a = RRTT ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vart - rcm[a,CBET,c] * cvct
  a = RRUU ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varu - rcm[a,CBET,c] * cvcu
  a = RRVV ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varv - rcm[a,CBET,c] * cvcv
  a = RRWW ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varw - rcm[a,CBET,c] * cvcw
  a = RRCC ; rcm[a,CLIN,c] =                                          100
             rcm[a,CNOL,c] =                                          0.0
             rcm[a,CNOA,c] =                                          0.0
  a = RRUU ; rcm[a,CLIN,c] = rcm[a,CBET,c]^2 * rcm[a,CTRU,c] / varu * 100
             rcm[a,CNOL,c] =                                          0.0
             rcm[a,CNOA,c] =                   rcm[a,CERI,c] / varu * 100
  c = MRLR
  a = RRCC ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] =         1.0 ; rcm[a,CALP,c] =                         0.0
  a = RRSS ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvsu / cvcu ; rcm[a,CALP,c] = avgs - rcm[a,CBET,c] * avgc
  a = RRTT ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvtu / cvcu ; rcm[a,CALP,c] = avgt - rcm[a,CBET,c] * avgc
  a = RRUU ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = varu / cvcu ; rcm[a,CALP,c] = avgu - rcm[a,CBET,c] * avgc
  a = RRVV ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvuv / cvcu ; rcm[a,CALP,c] = avgv - rcm[a,CBET,c] * avgc
  a = RRWW ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvuw / cvcu ; rcm[a,CALP,c] = avgw - rcm[a,CBET,c] * avgc
  a = RRCC ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varc - cvcu * cvcu / varu
  a = RRSS ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vars - cvsu * cvsu / varu
  a = RRTT ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vart - cvtu * cvtu / varu
  a = RRUU ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] =                       0.0
  a = RRVV ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varv - cvuv * cvuv / varu
  a = RRWW ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varw - cvuw * cvuw / varu
  a = RRCC ; rcm[a,CLIN,c] = rcm[a,CTRU,c] / varc * 100
             rcm[a,CNOL,c] =                        0.0
             rcm[a,CNOA,c] = rcm[a,CERI,c] / varc * 100
  a = RRUU ; rcm[a,CLIN,c] =                        100
             rcm[a,CNOL,c] =                        0.0
             rcm[a,CNOA,c] =                        0.0

  function weak(tttt::Float64, betu::Float64)                                 # provide weak solution constraints
    mskid = 0                                                                 # and a negative covariance mask
    ccBT = varc -          tttt                        ; ccBT <= 0 && (mskid += 1)
    cuBT = cvcu - betu   * tttt                        ; cuBT <= 0 && (mskid += 10)
    uuBT = varu - betu^2 * tttt                        ; uuBT <= 0 && (mskid += 100)
    lams = (betu * cvcs - cvsu) / (betu * cvct - cvtu) ; lams <  0 && (mskid += 2)    #; lams > 1 && (mskid += 4)
    lamt = (betu * cvct - cvtu) / (betu * cuBT - uuBT) ; lamt <  0 && (mskid += 20)   #; lamt > 1 && (mskid += 40)
    lamv = (betu * cvcv - cvuv) / (betu * cuBT - uuBT) ; lamv <  0 && (mskid += 200)  #; lamv > 1 && (mskid += 400)
    lamw = (betu * cvcw - cvuw) / (betu * cvcv - cvuv) ; lamw <  0 && (mskid += 2000) #; lamw > 1 && (mskid += 4000)
    bets = (cvcs - lams * lamt * cuBT) / tttt
    bett = (cvct -        lamt * cuBT) / tttt
    betv = (cvcv - lamv *        cuBT) / tttt
    betw = (cvcw - lamv * lamw * cuBT) / tttt
    ssBT = vars -        bets^2 * tttt                 ; ssBT <= 0 && (mskid += 10000)
    ttBT = vart -        bett^2 * tttt                 ; ttBT <= 0 && (mskid += 100000)
    vvBT = varv -        betv^2 * tttt                 ; vvBT <= 0 && (mskid += 1000000)
    wwBT = varw -        betw^2 * tttt                 ; wwBT <= 0 && (mskid += 10000000)
#   csBT = cvcs -        bets   * tttt                 ; csBT <= 0 && (mskid += 20000)
#   ctBT = cvct -        bett   * tttt                 ; ctBT <= 0 && (mskid += 200000)
#   cvBT = cvcv -        betv   * tttt                 ; cvBT <= 0 && (mskid += 2000000)
#   cwBT = cvcw -        betw   * tttt                 ; cwBT <= 0 && (mskid += 20000000)
    eecc = ccBT -          cuBT                        ; eecc <= 0 && (mskid += 100000000)
    eess = ssBT - lams^2 * ttBT                        ; eess <= 0 && (mskid += 1000000000)
    eett = ttBT - lamt^2 * uuBT                        ; eett <= 0 && (mskid += 10000000000)
    eeuu = uuBT -          cuBT                        ; eeuu <= 0 && (mskid += 100000000000)
    eevv = vvBT - lamv^2 * uuBT                        ; eevv <= 0 && (mskid += 1000000000000)
    eeww = wwBT - lamw^2 * vvBT                        ; eeww <= 0 && (mskid += 10000000000000)

    wkst = abs(cvst - bets * bett * tttt - lams *                      ttBT)  # (weak constraints are just the covariance
    wktv = abs(cvtv - bett * betv * tttt -        lamt * lamv *        uuBT)  # eqns that exclude those involving C and U)
    wkvw = abs(cvvw - betv * betw * tttt -                      lamw * vvBT)
    wksv = abs(cvsv - bets * betv * tttt - lams * lamt * lamv *        uuBT)
    wksw = abs(cvsw - bets * betw * tttt - lams * lamt * lamv * lamw * uuBT)
    wktw = abs(cvtw - bett * betw * tttt -        lamt * lamv * lamw * uuBT)
    wkto =           (wkst +     wktv +     wkvw +     wksv +     wksw +     wktw) / 6
    return(mskid, log(wkst), log(wktv), log(wkvw), log(wksv), log(wksw), log(wktw), log(wkto))
  end

  solve = true                                                                # search for positive-variance EIV solutions
  mintt =  0.0                                                                # that are bounded by OLR and RLR and a wide
  maxtt =  2.0 * varc                                                         # range of shared true variance (tttt) values
  minbu = cvcu / varc                                                         # (and allow that there might be no solution)
  maxbu = varu / cvcu
  minbu > maxbu && ((minbu, maxbu) = (maxbu, minbu))
  rngtt = collect(range(mintt, stop = maxtt, length = ESTIM + 2))[2:end-1]
  rngbu = collect(range(minbu, stop = maxbu, length = ESTIM + 2))[2:end-1]
  est00 = Array{Float64}(undef, ESTIM, ESTIM)
  est01 = Array{Float64}(undef, ESTIM, ESTIM)
  est02 = Array{Float64}(undef, ESTIM, ESTIM)
  est03 = Array{Float64}(undef, ESTIM, ESTIM)
  est04 = Array{Float64}(undef, ESTIM, ESTIM)
  est05 = Array{Float64}(undef, ESTIM, ESTIM)
  est06 = Array{Float64}(undef, ESTIM, ESTIM)
  est99 = Array{Float64}(undef, ESTIM, ESTIM)
  for (a, vala) in enumerate(rngtt), (b, valb) in enumerate(rngbu)
    est00[a,b], est01[a,b], est02[a,b], est03[a,b], est04[a,b], est05[a,b], est06[a,b], est99[a,b] = weak(vala, valb)
  end
  !any(x -> x == 0, est00) && (solve = false)                                 # find a positive-variance consensus-path
  !solve && print("\ncpseiv ERROR : no positive variance solution\n\n")       # solution by covariances that exclude CU

  solve, tarit, tarib, finit, finib, esttt, estbu, smoopass, msk01, msk02, msk03, msk04, msk05, msk06 = consensus(solve, rngtt, rngbu, est00, est01, est02, est03, est04, est05, est06)
  tttt = avg[PFTT] = rngtt[finit] ; betu = avg[PFBB] = rngbu[finib]
         avg[PTTT] = rngtt[tarit] ;        avg[PTBB] = rngbu[tarib]
         avg[PDIS] = (((finit - tarit)^2 + (finib - tarib)^2) / (2 * ESTIM^2))^0.5

  ccBT = varc -          tttt                                                 # derive the EIV metrics and complete the RCM
  cuBT = cvcu - betu   * tttt
  uuBT = varu - betu^2 * tttt
  lams = (betu * cvcs - cvsu) / (betu * cvct - cvtu) ; bets = (cvcs - lams * lamt * cuBT) / tttt
  lamt = (betu * cvct - cvtu) / (betu * cuBT - uuBT) ; bett = (cvct -        lamt * cuBT) / tttt
  lamv = (betu * cvcv - cvuv) / (betu * cuBT - uuBT) ; betv = (cvcv - lamv *        cuBT) / tttt
  lamw = (betu * cvcw - cvuw) / (betu * cvcv - cvuv) ; betw = (cvcw - lamv * lamw * cuBT) / tttt
  ssBT = vars - bets^2 * tttt
  ttBT = vart - bett^2 * tttt
  vvBT = varv - betv^2 * tttt
  wwBT = varw - betw^2 * tttt
  eecc = ccBT -          cuBT
  eess = ssBT - lams^2 * ttBT
  eett = ttBT - lamt^2 * uuBT
  eeuu = uuBT -          cuBT
  eevv = vvBT - lamv^2 * uuBT
  eeww = wwBT - lamw^2 * vvBT

  c = MEIV
  a = RRCC ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] =  1.0 ; rcm[a,CALP,c] =                0.0
  a = RRSS ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = bets ; rcm[a,CALP,c] = avgs - bets * avgc
  a = RRTT ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = bett ; rcm[a,CALP,c] = avgt - bett * avgc
  a = RRUU ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betu ; rcm[a,CALP,c] = avgu - betu * avgc
  a = RRVV ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betv ; rcm[a,CALP,c] = avgv - betv * avgc
  a = RRWW ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betw ; rcm[a,CALP,c] = avgw - betw * avgc
  a = RRCC ; rcm[a,CLAM,c] =  0.0 ; rcm[a,CERI,c] = eecc ; rcm[a,CERT,c] = ccBT
  a = RRSS ; rcm[a,CLAM,c] = lams ; rcm[a,CERI,c] = eess ; rcm[a,CERT,c] = ssBT
  a = RRTT ; rcm[a,CLAM,c] = lamt ; rcm[a,CERI,c] = eett ; rcm[a,CERT,c] = ttBT
  a = RRUU ; rcm[a,CLAM,c] =  0.0 ; rcm[a,CERI,c] = eeuu ; rcm[a,CERT,c] = uuBT
  a = RRVV ; rcm[a,CLAM,c] = lamv ; rcm[a,CERI,c] = eevv ; rcm[a,CERT,c] = vvBT
  a = RRWW ; rcm[a,CLAM,c] = lamw ; rcm[a,CERI,c] = eeww ; rcm[a,CERT,c] = wwBT
  a = RRCC ; rcm[a,CLIN,c] =          tttt / varc * 100 ; rcm[a,CNOL,c] = cuBT / varc * 100 ; rcm[a,CNOA,c] = eecc / varc * 100
  a = RRUU ; rcm[a,CLIN,c] = betu^2 * tttt / varu * 100 ; rcm[a,CNOL,c] = cuBT / varu * 100 ; rcm[a,CNOA,c] = eeuu / varu * 100

  sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", rcm[RRCC,CLIN,c], rcm[RRCC,CNOL,c], rcm[RRCC,CNOA,c],
                                                                    rcm[RRUU,CLIN,c], rcm[RRUU,CNOL,c], rcm[RRUU,CNOA,c])

  if pic != ""                                                                # then plot the EIV solution
    if last(picrng) > first(picrng)
      intpic = collect(picrng)[2:end] .- 0.5 * step(picrng) ; lenpic = length(intpic)
      fil01 = pic * ".hst01.nc" ; isfile(fil01) && rm(fil01) ; nccreer(fil01, 1, [0.0], intpic, missval)
      fil02 = pic * ".hst02.nc" ; isfile(fil02) && rm(fil02) ; nccreer(fil02, 1, [0.0], intpic, missval)
      fil03 = pic * ".hst03.nc" ; isfile(fil03) && rm(fil03) ; nccreer(fil03, 1, [0.0], intpic, missval)
      fil04 = pic * ".hst04.nc" ; isfile(fil04) && rm(fil04) ; nccreer(fil04, 1, [0.0], intpic, missval)
      fil05 = pic * ".hst05.nc" ; isfile(fil05) && rm(fil05) ; nccreer(fil05, 1, [0.0], intpic, missval)
      fil06 = pic * ".hst06.nc" ; isfile(fil06) && rm(fil06) ; nccreer(fil06, 1, [0.0], intpic, missval)
      fil51 = pic * ".hst51.nc" ; isfile(fil51) && rm(fil51) ; nccreer(fil51, 1, [0.0], intpic, missval)
      fil52 = pic * ".hst52.nc" ; isfile(fil52) && rm(fil52) ; nccreer(fil52, 1, [0.0], intpic, missval)
      fil53 = pic * ".hst53.nc" ; isfile(fil53) && rm(fil53) ; nccreer(fil53, 1, [0.0], intpic, missval)
      fil54 = pic * ".hst54.nc" ; isfile(fil54) && rm(fil54) ; nccreer(fil54, 1, [0.0], intpic, missval)
      fil55 = pic * ".hst55.nc" ; isfile(fil55) && rm(fil55) ; nccreer(fil55, 1, [0.0], intpic, missval)
      fil56 = pic * ".hst56.nc" ; isfile(fil56) && rm(fil56) ; nccreer(fil56, 1, [0.0], intpic, missval)
      ncwrite(logfithist(cc[  mask], picrng), fil01, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(ss[  mask], picrng), fil02, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(tt[  mask], picrng), fil03, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(uu[  mask], picrng), fil04, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(vv[  mask], picrng), fil05, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(ww[  mask], picrng), fil06, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(cc[.!mask], picrng), fil51, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(ss[.!mask], picrng), fil52, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(tt[.!mask], picrng), fil53, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(uu[.!mask], picrng), fil54, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(vv[.!mask], picrng), fil55, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(ww[.!mask], picrng), fil56, "tmp", start=[1,1,1], count=[lenpic,1,1])
      filaa = pic * ".txt" ; lena = length(cc[mask]) ; lenb = length(cc[.!mask]) ; lenc = lena + lenb
      line = @sprintf("%d %d %7.1f %7.1f %9.5f %.0f %7.2f\n", lena, lenb, 100 * lenb / lenc, 100 * limmcd, ecdfsup(cc[mask], uu[mask]), difhist(cc[mask], uu[mask], picrng), precalbet)
      fpa = ouvre(filaa, "w")          ; write(fpa,     line)
      write(fpa, statline(cc[  mask])) ; write(fpa, statline(uu[  mask])) ; write(fpa, statline(ss[  mask]))
      write(fpa, statline(tt[  mask])) ; write(fpa, statline(vv[  mask])) ; write(fpa, statline(ww[  mask]))
      write(fpa, statline(cc[.!mask])) ; write(fpa, statline(uu[.!mask])) ; write(fpa, statline(ss[.!mask]))
      write(fpa, statline(tt[.!mask])) ; write(fpa, statline(vv[.!mask])) ; write(fpa, statline(ww[.!mask]))
      write(fpa, "Calibrated\n")       ; write(fpa, "Uncalibrat\n")       ; write(fpa, "Uncal (T-2)\n")
      write(fpa, "Uncal (T-1)\n")      ; write(fpa, "Uncal (T+1)\n")      ; write(fpa, "Uncal (T+2)\n")
      close(fpa)
      if PLOTPROG
        print("grads --quiet -blc \"ErrorsInVariables.plot.distribution $fil01 $fil04 $fil02 $fil03 $fil05 $fil06 $fil51 $fil54 $fil52 $fil53 $fil55 $fil56 $filaa $pic.dist\"\n")
          run(`grads --quiet -blc  "ErrorsInVariables.plot.distribution $fil01 $fil04 $fil02 $fil03 $fil05 $fil06 $fil51 $fil54 $fil52 $fil53 $fil55 $fil56 $filaa $pic.dist"`)
      end
      !keepnc && (rm(fil01) ; rm(fil02) ; rm(fil03) ; rm(fil04) ; rm(fil05) ; rm(fil06) ; rm(fil51) ; rm(fil52) ; rm(fil53) ; rm(fil54) ; rm(fil55) ; rm(fil56) ; rm(filaa))

      filbb = pic * ".hstin.nc" ; isfile(filbb) && rm(filbb) ; nccreer(filbb, 1, intpic, intpic, missval)
      ncwrite(logfitdoub(cc[mask], uu[mask], picrng), filbb, "tmp", start=[1,1,1], count=[lenpic,lenpic,1])
      filcc = pic * ".txt" ; rcmsave(avg, rcm, filcc; sumlin = sumlin)
      if PLOTPROG
        print("grads --quiet -blc \"ErrorsInVariables.plot.doublebution $filbb $filcc $pic.doub\"\n")
          run(`grads --quiet -blc  "ErrorsInVariables.plot.doublebution $filbb $filcc $pic.doub"`)
      end
      !keepnc && (rm(filbb) ; rm(filcc * ".IAVG") ; rm(filcc * ".MOLR") ; rm(filcc * ".MEIV") ; rm(filcc * ".MRLR"))
    end

    for (a, vala) in enumerate(rngtt), (b, valb) in enumerate(rngbu)
      est01[a,b] > CUTOFF && (est01[a,b] = CUTOFF)
      est02[a,b] > CUTOFF && (est02[a,b] = CUTOFF)
      est03[a,b] > CUTOFF && (est03[a,b] = CUTOFF)
      est04[a,b] > CUTOFF && (est04[a,b] = CUTOFF)
      est05[a,b] > CUTOFF && (est05[a,b] = CUTOFF)
      est06[a,b] > CUTOFF && (est06[a,b] = CUTOFF)
    end
    fil00 = pic * ".est00.nc" ; isfile(fil00) && rm(fil00) ; nccreer(fil00, 1, rngbu, rngtt, missval)
    fil01 = pic * ".est01.nc" ; isfile(fil01) && rm(fil01) ; nccreer(fil01, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil02 = pic * ".est02.nc" ; isfile(fil02) && rm(fil02) ; nccreer(fil02, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil03 = pic * ".est03.nc" ; isfile(fil03) && rm(fil03) ; nccreer(fil03, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil04 = pic * ".est04.nc" ; isfile(fil04) && rm(fil04) ; nccreer(fil04, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil05 = pic * ".est05.nc" ; isfile(fil05) && rm(fil05) ; nccreer(fil05, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil06 = pic * ".est06.nc" ; isfile(fil06) && rm(fil06) ; nccreer(fil06, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil99 = pic * ".est99.nc" ; isfile(fil99) && rm(fil99) ; nccreer(fil99, 1, rngbu, rngtt, missval)
    filtt = pic * ".esttt.nc" ; isfile(filtt) && rm(filtt) ; nccreer(filtt, 1, rngbu, rngtt, missval)
    filbu = pic * ".estbu.nc" ; isfile(filbu) && rm(filbu) ; nccreer(filbu, 1, rngbu, rngtt, missval)
            est00[finit,finib] = -1.0
    ncwrite(est00, fil00, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est01, fil01, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk01), fil01, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est02, fil02, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk02), fil02, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est03, fil03, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk03), fil03, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est04, fil04, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk04), fil04, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est05, fil05, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk05), fil05, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est06, fil06, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk06), fil06, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est99, fil99, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(esttt, filtt, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(estbu, filbu, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    fildd = pic * ".estot.txt" ; line = @sprintf("%f %d %d %f %f %f %d %d %f %f %f %d\n", 0.0, tarit, tarib, avg[PTTT], avg[PTBB], est99[tarit,tarib], finit, finib, avg[PFTT], avg[PFBB], est99[finit,finib], smoopass)
    fpa = ouvre(fildd, "w")  ; write(fpa, line)
    write(fpa, "Cov(S,T)\n") ; write(fpa, "Cov(T,V)\n") ; write(fpa, "Cov(V,W)\n")
    write(fpa, "Cov(S,V)\n") ; write(fpa, "Cov(S,W)\n") ; write(fpa, "Cov(T,W)\n")
    close(fpa)
    if PLOTPROG
      print("grads --quiet -blc \"ErrorsInVariables.plot.solution $fil00 $fil01 $fil02 $fil03 $fil04 $fil05 $fil06 $fil99 $fildd $filtt $filbu $pic\"\n")
        run(`grads --quiet -blc  "ErrorsInVariables.plot.solution $fil00 $fil01 $fil02 $fil03 $fil04 $fil05 $fil06 $fil99 $fildd $filtt $filbu $pic"`)
    end
    !keepnc && (rm(fil00) ; rm(fil01) ; rm(fil02) ; rm(fil03) ; rm(fil04) ; rm(fil05) ; rm(fil06) ; rm(fil99) ; rm(filtt) ; rm(filbu) ; rm(fildd))
  end

  if echotxt != [] && solve != false
    @printf("\nnumber of collocations including outliers = %15d\n", length(cc))
    @printf(  "number of collocations excluding outliers = %15d\n", length(cc[mask]))
    for a in echotxt
      @printf("\nrcm[%d,CTRU,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CTRU,MOLR], rcm[a,CTRU,MEIV], rcm[a,CTRU,MRLR])
      @printf(  "rcm[%d,CALP,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CALP,MOLR], rcm[a,CALP,MEIV], rcm[a,CALP,MRLR])
      @printf(  "rcm[%d,CBET,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CBET,MOLR], rcm[a,CBET,MEIV], rcm[a,CBET,MRLR])
      @printf(  "rcm[%d,CLAM,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CLAM,MOLR], rcm[a,CLAM,MEIV], rcm[a,CLAM,MRLR])
      @printf(  "rcm[%d,CERI,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CERI,MOLR], rcm[a,CERI,MEIV], rcm[a,CERI,MRLR])
      @printf(  "rcm[%d,CERT,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CERT,MOLR], rcm[a,CERT,MEIV], rcm[a,CERT,MRLR])
    end
    @printf("\navg[RRCC]           = %15.8f\n",               avg[RRCC])
    @printf(  "avg[RRUU]           = %15.8f\n",               avg[RRUU])
    @printf(  "avg[PDIS]           = %15.8f\n\n",             avg[PDIS])
    @printf("           Alpha            Beta    VAR(I) Linear       Nonlinear    Unassociated   VAR(N) Linear       Nonlinear    Unassociated\n")
    @printf(" %15.8f %15.8f %s\n", precalalp, precalbet, sumlin)
  end
  solve == false && print("\ncpseiv ERROR : returning a missing solution\n\n")
  solve == false && (rcm[:,:,MEIV] .= missval ; sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", missval, missval, missval, missval, missval, missval))
  return(avg, rcm, sumlin)
end

function cpseiv(bb::Array{Float64,1}, cc::Array{Float64,1}, dd::Array{Float64,1}, precalalp::Float64, precalbet::Float64, tt::Array{Float64,1}, uu::Array{Float64,1}, vv::Array{Float64,1}; missval = MISS, detmcd = true, limmcd = MCDTRIM, pic = "", picrng = 1.0:0.0, keepnc = false, echotxt = [])
  avg    = fill(missval, PDIS)
  rcm    = fill(missval, (RRCM, CRCM, MRCM))
  sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", missval, missval, missval, missval, missval, missval)
  length(cc) < 50 && return(avg, rcm, sumlin)

# if detmcd                                                                   # get a mask to exclude outliers using DetMCD in R
#   temp = [bb cc dd tt uu vv]                                                # (before precalibration, avgc = remp[:center][2],
#   remp = rcopy(R"DetMCD($temp, alpha = $limmcd)")                           # varc = remp[:cov][2,2], cvcu = remp[:cov][2,5])
#   mask = falses(length(cc)) ; for a in remp[:Hsubsets]  mask[a] = true  end
# else
    mask =  trues(length(cc))
# end

  if precalalp == MISS && precalbet == MISS
    avgc = mean(cc[mask]) ; varc = var(cc[mask])
    avgu = mean(uu[mask]) ; varu = var(uu[mask])
    cvcu = cov(cc[mask], uu[mask])
    precalbet = sign(cvcu) * (varu / varc)^0.5
    precalalp = avgu - precalbet * avgc
    @printf("\npreliminary calibration by variance matchng alp = %15.8f bet = %15.8f\n", precalalp, precalbet)
  else
    @printf("\npreliminary calibration by specified values alp = %15.8f bet = %15.8f\n", precalalp, precalbet)
  end
  for a = 1:length(uu)
    tt[a] -= precalalp ; tt[a] /= precalbet
    uu[a] -= precalalp ; uu[a] /= precalbet
    vv[a] -= precalalp ; vv[a] /= precalbet
  end

  avgb = mean(bb[mask]) ; varb = var(bb[mask])
  avgc = mean(cc[mask]) ; varc = var(cc[mask])
  avgd = mean(dd[mask]) ; vard = var(dd[mask])
  avgt = mean(tt[mask]) ; vart = var(tt[mask])
  avgu = mean(uu[mask]) ; varu = var(uu[mask])
  avgv = mean(vv[mask]) ; varv = var(vv[mask])
  cvbc = cov(bb[mask], cc[mask])
  cvbd = cov(bb[mask], dd[mask])
  cvcd = cov(cc[mask], dd[mask])
  cvtu = cov(tt[mask], uu[mask])
  cvtv = cov(tt[mask], vv[mask])
  cvuv = cov(uu[mask], vv[mask])
  cvbt = cov(bb[mask], tt[mask])
  cvbu = cov(bb[mask], uu[mask])
  cvbv = cov(bb[mask], vv[mask])
  cvct = cov(cc[mask], tt[mask])
  cvcu = cov(cc[mask], uu[mask])
  cvcv = cov(cc[mask], vv[mask])                                              # get RCM OLR (no C error) and RLR (no U error) values
  cvdt = cov(dd[mask], tt[mask])                                              # for available predictive samples among ABCDE and STUVW
  cvdu = cov(dd[mask], uu[mask])                                              # (for simplicity, CLAM = 0 for OLR-STVW and RLR-ABDE)
  cvdv = cov(dd[mask], vv[mask])
  avg = [MISS, avgb, avgc, avgd, MISS, MISS, avgt, avgu, avgv, MISS, length(cc[mask]), precalalp, precalbet, MISS, MISS, MISS, MISS, MISS]

  rcm = zeros(RRCM, CRCM, MRCM)
  a = RRBB ; rcm[a,CCVB,:] .= varb ; rcm[a,CCVC,:] .= cvbc ; rcm[a,CCVD,:] .= cvbd ; rcm[a,CCVT,:] .= cvbt ; rcm[a,CCVU,:] .= cvbu ; rcm[a,CCVV,:] .= cvbv
  a = RRCC ; rcm[a,CCVB,:] .= cvbc ; rcm[a,CCVC,:] .= varc ; rcm[a,CCVD,:] .= cvcd ; rcm[a,CCVT,:] .= cvct ; rcm[a,CCVU,:] .= cvcu ; rcm[a,CCVV,:] .= cvcv
  a = RRDD ; rcm[a,CCVB,:] .= cvbd ; rcm[a,CCVC,:] .= cvcd ; rcm[a,CCVD,:] .= vard ; rcm[a,CCVT,:] .= cvdt ; rcm[a,CCVU,:] .= cvdu ; rcm[a,CCVV,:] .= cvdv
  a = RRTT ; rcm[a,CCVB,:] .= cvbt ; rcm[a,CCVC,:] .= cvct ; rcm[a,CCVD,:] .= cvdt ; rcm[a,CCVT,:] .= vart ; rcm[a,CCVU,:] .= cvtu ; rcm[a,CCVV,:] .= cvtv
  a = RRUU ; rcm[a,CCVB,:] .= cvbu ; rcm[a,CCVC,:] .= cvcu ; rcm[a,CCVD,:] .= cvdu ; rcm[a,CCVT,:] .= cvtu ; rcm[a,CCVU,:] .= varu ; rcm[a,CCVV,:] .= cvuv
  a = RRVV ; rcm[a,CCVB,:] .= cvbv ; rcm[a,CCVC,:] .= cvcv ; rcm[a,CCVD,:] .= cvdv ; rcm[a,CCVT,:] .= cvtv ; rcm[a,CCVU,:] .= cvuv ; rcm[a,CCVV,:] .= varv
  c = MOLR
  a = RRBB ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvbc / varc ; rcm[a,CALP,c] = avgb - rcm[a,CBET,c] * avgc
  a = RRCC ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] =         1.0 ; rcm[a,CALP,c] =                         0.0
  a = RRDD ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcd / varc ; rcm[a,CALP,c] = avgd - rcm[a,CBET,c] * avgc
  a = RRTT ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvct / varc ; rcm[a,CALP,c] = avgt - rcm[a,CBET,c] * avgc
  a = RRUU ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcu / varc ; rcm[a,CALP,c] = avgu - rcm[a,CBET,c] * avgc
  a = RRVV ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcv / varc ; rcm[a,CALP,c] = avgv - rcm[a,CBET,c] * avgc
  a = RRBB ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varb - rcm[a,CBET,c] * cvbc
  a = RRCC ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] =                         0.0
  a = RRDD ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vard - rcm[a,CBET,c] * cvcd
  a = RRTT ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vart - rcm[a,CBET,c] * cvct
  a = RRUU ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varu - rcm[a,CBET,c] * cvcu
  a = RRVV ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varv - rcm[a,CBET,c] * cvcv
  a = RRCC ; rcm[a,CLIN,c] =                                          100
             rcm[a,CNOL,c] =                                          0.0
             rcm[a,CNOA,c] =                                          0.0
  a = RRUU ; rcm[a,CLIN,c] = rcm[a,CBET,c]^2 * rcm[a,CTRU,c] / varu * 100
             rcm[a,CNOL,c] =                                          0.0
             rcm[a,CNOA,c] =                   rcm[a,CERI,c] / varu * 100
  c = MRLR
  a = RRBB ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvbu / cvcu ; rcm[a,CALP,c] = avgb - rcm[a,CBET,c] * avgc
  a = RRCC ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] =         1.0 ; rcm[a,CALP,c] =                         0.0
  a = RRDD ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvdu / cvcu ; rcm[a,CALP,c] = avgd - rcm[a,CBET,c] * avgc
  a = RRTT ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvtu / cvcu ; rcm[a,CALP,c] = avgt - rcm[a,CBET,c] * avgc
  a = RRUU ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = varu / cvcu ; rcm[a,CALP,c] = avgu - rcm[a,CBET,c] * avgc
  a = RRVV ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvuv / cvcu ; rcm[a,CALP,c] = avgv - rcm[a,CBET,c] * avgc
  a = RRBB ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varb - cvbu * cvbu / varu
  a = RRCC ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varc - cvcu * cvcu / varu
  a = RRDD ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vard - cvdu * cvdu / varu
  a = RRTT ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vart - cvtu * cvtu / varu
  a = RRUU ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] =                       0.0
  a = RRVV ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varv - cvuv * cvuv / varu
  a = RRCC ; rcm[a,CLIN,c] = rcm[a,CTRU,c] / varc * 100
             rcm[a,CNOL,c] =                        0.0
             rcm[a,CNOA,c] = rcm[a,CERI,c] / varc * 100
  a = RRUU ; rcm[a,CLIN,c] =                        100
             rcm[a,CNOL,c] =                        0.0
             rcm[a,CNOA,c] =                        0.0

  function weak(tttt::Float64, betu::Float64)                                 # provide weak solution constraints
    mskid = 0                                                                 # and a negative covariance mask
    ccBT = varc -          tttt                        ; ccBT <= 0 && (mskid += 1)
    cuBT = cvcu - betu   * tttt                        ; cuBT <= 0 && (mskid += 10)
    uuBT = varu - betu^2 * tttt                        ; uuBT <= 0 && (mskid += 100)
    lamb = (betu * cvbc - cvbu) / (betu * ccBT - cuBT) ; lamb <  0 && (mskid += 2)    #; lamb > 1 && (mskid += 4)
    lamd = (betu * cvcd - cvdu) / (betu * ccBT - cuBT) ; lamd <  0 && (mskid += 20)   #; lamd > 1 && (mskid += 40)
    lamt = (betu * cvct - cvtu) / (betu * cuBT - uuBT) ; lamt <  0 && (mskid += 200)  #; lamt > 1 && (mskid += 400)
    lamv = (betu * cvcv - cvuv) / (betu * cuBT - uuBT) ; lamv <  0 && (mskid += 2000) #; lamv > 1 && (mskid += 4000)
    betb = (cvbc - lamb * ccBT) /         tttt
    betd = (cvcd - lamd * ccBT) /         tttt
    bett = (cvtu - lamt * uuBT) / (betu * tttt)
    betv = (cvuv - lamv * uuBT) / (betu * tttt)
    bbBT = varb -        betb^2 * tttt                 ; bbBT <= 0 && (mskid += 10000)
    ddBT = vard -        betd^2 * tttt                 ; ddBT <= 0 && (mskid += 100000)
    ttBT = vart -        bett^2 * tttt                 ; ttBT <= 0 && (mskid += 1000000)
    vvBT = varv -        betv^2 * tttt                 ; vvBT <= 0 && (mskid += 10000000)
#   bcBT = cvbc -        betb   * tttt                 ; bcBT <= 0 && (mskid += 20000)
#   cdBT = cvcd -        betd   * tttt                 ; cdBT <= 0 && (mskid += 200000)
#   tuBT = cvtu - betu * bett   * tttt                 ; tuBT <= 0 && (mskid += 2000000)
#   uvBT = cvuv - betu * betv   * tttt                 ; uvBT <= 0 && (mskid += 20000000)
    eebb = bbBT - lamb^2 * ccBT                        ; eebb <= 0 && (mskid += 100000000)
    eecc = ccBT -          cuBT                        ; eecc <= 0 && (mskid += 1000000000)
    eedd = ddBT - lamd^2 * ccBT                        ; eedd <= 0 && (mskid += 10000000000)
    eett = ttBT - lamt^2 * uuBT                        ; eett <= 0 && (mskid += 100000000000)
    eeuu = uuBT -          cuBT                        ; eeuu <= 0 && (mskid += 1000000000000)
    eevv = vvBT - lamv^2 * uuBT                        ; eevv <= 0 && (mskid += 10000000000000)

    wkbt = abs(cvbt - betb * bett * tttt - lamb * lamt * cuBT)                # (weak constraints are just the covariance
    wkbd = abs(cvbd - betb * betd * tttt - lamb * lamd * ccBT)                # eqns that exclude those involving C and U)
    wkbv = abs(cvbv - betb * betv * tttt - lamb * lamv * cuBT)
    wkdt = abs(cvdt - betd * bett * tttt - lamd * lamt * cuBT)
    wktv = abs(cvtv - bett * betv * tttt - lamt * lamv * uuBT)
    wkdv = abs(cvdv - betd * betv * tttt - lamd * lamv * cuBT)
    wkto =           (wkbt +     wkbd +     wkbv +     wkdt +     wktv +     wkdv) / 6
    return(mskid, log(wkbt), log(wkbd), log(wkbv), log(wkdt), log(wktv), log(wkdv), log(wkto))
  end

  solve = true                                                                # search for positive-variance EIV solutions
  mintt =  0.0                                                                # that are bounded by OLR and RLR and a wide
  maxtt =  2.0 * varc                                                         # range of shared true variance (tttt) values
  minbu = cvcu / varc                                                         # (and allow that there might be no solution)
  maxbu = varu / cvcu
  minbu > maxbu && ((minbu, maxbu) = (maxbu, minbu))
  rngtt = collect(range(mintt, stop = maxtt, length = ESTIM + 2))[2:end-1]
  rngbu = collect(range(minbu, stop = maxbu, length = ESTIM + 2))[2:end-1]
  est00 = Array{Float64}(undef, ESTIM, ESTIM)
  est01 = Array{Float64}(undef, ESTIM, ESTIM)
  est02 = Array{Float64}(undef, ESTIM, ESTIM)
  est03 = Array{Float64}(undef, ESTIM, ESTIM)
  est04 = Array{Float64}(undef, ESTIM, ESTIM)
  est05 = Array{Float64}(undef, ESTIM, ESTIM)
  est06 = Array{Float64}(undef, ESTIM, ESTIM)
  est99 = Array{Float64}(undef, ESTIM, ESTIM)
  for (a, vala) in enumerate(rngtt), (b, valb) in enumerate(rngbu)
    est00[a,b], est01[a,b], est02[a,b], est03[a,b], est04[a,b], est05[a,b], est06[a,b], est99[a,b] = weak(vala, valb)
  end
  !any(x -> x == 0, est00) && (solve = false)                                 # find a positive-variance consensus-path
  !solve && print("\ncpseiv ERROR : no positive variance solution\n\n")       # solution by covariances that exclude CU

  solve, tarit, tarib, finit, finib, esttt, estbu, smoopass, msk01, msk02, msk03, msk04, msk05, msk06 = consensus(solve, rngtt, rngbu, est00, est01, est02, est03, est04, est05, est06)
  tttt = avg[PFTT] = rngtt[finit] ; betu = avg[PFBB] = rngbu[finib]
         avg[PTTT] = rngtt[tarit] ;        avg[PTBB] = rngbu[tarib]
         avg[PDIS] = (((finit - tarit)^2 + (finib - tarib)^2) / (2 * ESTIM^2))^0.5

  ccBT = varc -          tttt                                                 # derive the EIV metrics and complete the RCM
  cuBT = cvcu - betu   * tttt
  uuBT = varu - betu^2 * tttt
  lamb = (betu * cvbc - cvbu) / (betu * ccBT - cuBT) ; betb = (cvbc - lamb * ccBT) /         tttt
  lamd = (betu * cvcd - cvdu) / (betu * ccBT - cuBT) ; betd = (cvcd - lamd * ccBT) /         tttt
  lamt = (betu * cvct - cvtu) / (betu * cuBT - uuBT) ; bett = (cvtu - lamt * uuBT) / (betu * tttt)
  lamv = (betu * cvcv - cvuv) / (betu * cuBT - uuBT) ; betv = (cvuv - lamv * uuBT) / (betu * tttt)
  bbBT = varb -        betb^2 * tttt
  ddBT = vard -        betd^2 * tttt
  ttBT = vart -        bett^2 * tttt
  vvBT = varv -        betv^2 * tttt
  eebb = bbBT - lamb^2 * ccBT
  eecc = ccBT -          cuBT
  eedd = ddBT - lamd^2 * ccBT
  eett = ttBT - lamt^2 * uuBT
  eeuu = uuBT -          cuBT
  eevv = vvBT - lamv^2 * uuBT

  c = MEIV
  a = RRBB ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betb ; rcm[a,CALP,c] = avgb - betb * avgc
  a = RRCC ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] =  1.0 ; rcm[a,CALP,c] =                0.0
  a = RRDD ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betd ; rcm[a,CALP,c] = avgd - betd * avgc
  a = RRTT ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = bett ; rcm[a,CALP,c] = avgt - bett * avgc
  a = RRUU ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betu ; rcm[a,CALP,c] = avgu - betu * avgc
  a = RRVV ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betv ; rcm[a,CALP,c] = avgv - betv * avgc
  a = RRBB ; rcm[a,CLAM,c] = lamb ; rcm[a,CERI,c] = eebb ; rcm[a,CERT,c] = bbBT
  a = RRCC ; rcm[a,CLAM,c] =  0.0 ; rcm[a,CERI,c] = eecc ; rcm[a,CERT,c] = ccBT
  a = RRDD ; rcm[a,CLAM,c] = lamd ; rcm[a,CERI,c] = eedd ; rcm[a,CERT,c] = ddBT
  a = RRTT ; rcm[a,CLAM,c] = lamt ; rcm[a,CERI,c] = eett ; rcm[a,CERT,c] = ttBT
  a = RRUU ; rcm[a,CLAM,c] =  0.0 ; rcm[a,CERI,c] = eeuu ; rcm[a,CERT,c] = uuBT
  a = RRVV ; rcm[a,CLAM,c] = lamv ; rcm[a,CERI,c] = eevv ; rcm[a,CERT,c] = vvBT
  a = RRCC ; rcm[a,CLIN,c] =          tttt / varc * 100 ; rcm[a,CNOL,c] = cuBT / varc * 100 ; rcm[a,CNOA,c] = eecc / varc * 100
  a = RRUU ; rcm[a,CLIN,c] = betu^2 * tttt / varu * 100 ; rcm[a,CNOL,c] = cuBT / varu * 100 ; rcm[a,CNOA,c] = eeuu / varu * 100

  sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", rcm[RRCC,CLIN,c], rcm[RRCC,CNOL,c], rcm[RRCC,CNOA,c],
                                                                    rcm[RRUU,CLIN,c], rcm[RRUU,CNOL,c], rcm[RRUU,CNOA,c])

  if pic != ""                                                                # then plot the EIV solution
    if last(picrng) > first(picrng)
      intpic = collect(picrng)[2:end] .- 0.5 * step(picrng) ; lenpic = length(intpic)
      fil01 = pic * ".hst01.nc" ; isfile(fil01) && rm(fil01) ; nccreer(fil01, 1, [0.0], intpic, missval)
      fil02 = pic * ".hst02.nc" ; isfile(fil02) && rm(fil02) ; nccreer(fil02, 1, [0.0], intpic, missval)
      fil03 = pic * ".hst03.nc" ; isfile(fil03) && rm(fil03) ; nccreer(fil03, 1, [0.0], intpic, missval)
      fil04 = pic * ".hst04.nc" ; isfile(fil04) && rm(fil04) ; nccreer(fil04, 1, [0.0], intpic, missval)
      fil05 = pic * ".hst05.nc" ; isfile(fil05) && rm(fil05) ; nccreer(fil05, 1, [0.0], intpic, missval)
      fil06 = pic * ".hst06.nc" ; isfile(fil06) && rm(fil06) ; nccreer(fil06, 1, [0.0], intpic, missval)
      fil51 = pic * ".hst51.nc" ; isfile(fil51) && rm(fil51) ; nccreer(fil51, 1, [0.0], intpic, missval)
      fil52 = pic * ".hst52.nc" ; isfile(fil52) && rm(fil52) ; nccreer(fil52, 1, [0.0], intpic, missval)
      fil53 = pic * ".hst53.nc" ; isfile(fil53) && rm(fil53) ; nccreer(fil53, 1, [0.0], intpic, missval)
      fil54 = pic * ".hst54.nc" ; isfile(fil54) && rm(fil54) ; nccreer(fil54, 1, [0.0], intpic, missval)
      fil55 = pic * ".hst55.nc" ; isfile(fil55) && rm(fil55) ; nccreer(fil55, 1, [0.0], intpic, missval)
      fil56 = pic * ".hst56.nc" ; isfile(fil56) && rm(fil56) ; nccreer(fil56, 1, [0.0], intpic, missval)
      ncwrite(logfithist(bb[  mask], picrng), fil01, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(cc[  mask], picrng), fil02, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(dd[  mask], picrng), fil03, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(tt[  mask], picrng), fil04, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(uu[  mask], picrng), fil05, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(vv[  mask], picrng), fil06, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(bb[.!mask], picrng), fil51, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(cc[.!mask], picrng), fil52, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(dd[.!mask], picrng), fil53, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(tt[.!mask], picrng), fil54, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(uu[.!mask], picrng), fil55, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(vv[.!mask], picrng), fil56, "tmp", start=[1,1,1], count=[lenpic,1,1])
      filaa = pic * ".txt" ; lena = length(cc[mask]) ; lenb = length(cc[.!mask]) ; lenc = lena + lenb
      line = @sprintf("%d %d %7.1f %7.1f %9.5f %.0f %7.2f\n", lena, lenb, 100 * lenb / lenc, 100 * limmcd, ecdfsup(cc[mask], uu[mask]), difhist(cc[mask], uu[mask], picrng), precalbet)
      fpa = ouvre(filaa, "w")          ; write(fpa,     line)
      write(fpa, statline(cc[  mask])) ; write(fpa, statline(uu[  mask])) ; write(fpa, statline(bb[  mask]))
      write(fpa, statline(tt[  mask])) ; write(fpa, statline(dd[  mask])) ; write(fpa, statline(vv[  mask]))
      write(fpa, statline(cc[.!mask])) ; write(fpa, statline(uu[.!mask])) ; write(fpa, statline(bb[.!mask]))
      write(fpa, statline(tt[.!mask])) ; write(fpa, statline(dd[.!mask])) ; write(fpa, statline(vv[.!mask]))
      write(fpa, "Calibrated\n")       ; write(fpa, "Uncalibrat\n")       ; write(fpa, "Calib (T-1)\n")
      write(fpa, "Uncal (T-1)\n")      ; write(fpa, "Calib (T+1)\n")      ; write(fpa, "Uncal (T+1)\n")
      close(fpa)
      if PLOTPROG
        print("grads --quiet -blc \"ErrorsInVariables.plot.distribution $fil02 $fil05 $fil01 $fil04 $fil03 $fil06 $fil52 $fil55 $fil51 $fil54 $fil53 $fil56 $filaa $pic.dist\"\n")
          run(`grads --quiet -blc  "ErrorsInVariables.plot.distribution $fil02 $fil05 $fil01 $fil04 $fil03 $fil06 $fil52 $fil55 $fil51 $fil54 $fil53 $fil56 $filaa $pic.dist"`)
      end
      !keepnc && (rm(fil01) ; rm(fil02) ; rm(fil03) ; rm(fil04) ; rm(fil05) ; rm(fil06) ; rm(fil51) ; rm(fil52) ; rm(fil53) ; rm(fil54) ; rm(fil55) ; rm(fil56) ; rm(filaa))

      filbb = pic * ".hstin.nc" ; isfile(filbb) && rm(filbb) ; nccreer(filbb, 1, intpic, intpic, missval)
      ncwrite(logfitdoub(cc[mask], uu[mask], picrng), filbb, "tmp", start=[1,1,1], count=[lenpic,lenpic,1])
      filcc = pic * ".txt" ; rcmsave(avg, rcm, filcc; sumlin = sumlin)
      if PLOTPROG
        print("grads --quiet -blc \"ErrorsInVariables.plot.doublebution $filbb $filcc $pic.doub\"\n")
          run(`grads --quiet -blc  "ErrorsInVariables.plot.doublebution $filbb $filcc $pic.doub"`)
      end
      !keepnc && (rm(filbb) ; rm(filcc * ".IAVG") ; rm(filcc * ".MOLR") ; rm(filcc * ".MEIV") ; rm(filcc * ".MRLR"))
    end

    for (a, vala) in enumerate(rngtt), (b, valb) in enumerate(rngbu)
      est01[a,b] > CUTOFF && (est01[a,b] = CUTOFF)
      est02[a,b] > CUTOFF && (est02[a,b] = CUTOFF)
      est03[a,b] > CUTOFF && (est03[a,b] = CUTOFF)
      est04[a,b] > CUTOFF && (est04[a,b] = CUTOFF)
      est05[a,b] > CUTOFF && (est05[a,b] = CUTOFF)
      est06[a,b] > CUTOFF && (est06[a,b] = CUTOFF)
    end
    fil00 = pic * ".est00.nc" ; isfile(fil00) && rm(fil00) ; nccreer(fil00, 1, rngbu, rngtt, missval)
    fil01 = pic * ".est01.nc" ; isfile(fil01) && rm(fil01) ; nccreer(fil01, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil02 = pic * ".est02.nc" ; isfile(fil02) && rm(fil02) ; nccreer(fil02, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil03 = pic * ".est03.nc" ; isfile(fil03) && rm(fil03) ; nccreer(fil03, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil04 = pic * ".est04.nc" ; isfile(fil04) && rm(fil04) ; nccreer(fil04, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil05 = pic * ".est05.nc" ; isfile(fil05) && rm(fil05) ; nccreer(fil05, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil06 = pic * ".est06.nc" ; isfile(fil06) && rm(fil06) ; nccreer(fil06, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil99 = pic * ".est99.nc" ; isfile(fil99) && rm(fil99) ; nccreer(fil99, 1, rngbu, rngtt, missval)
    filtt = pic * ".esttt.nc" ; isfile(filtt) && rm(filtt) ; nccreer(filtt, 1, rngbu, rngtt, missval)
    filbu = pic * ".estbu.nc" ; isfile(filbu) && rm(filbu) ; nccreer(filbu, 1, rngbu, rngtt, missval)
            est00[finit,finib] = -1.0
    ncwrite(est00, fil00, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est01, fil01, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk01), fil01, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est02, fil02, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk02), fil02, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est03, fil03, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk03), fil03, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est04, fil04, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk04), fil04, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est05, fil05, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk05), fil05, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est06, fil06, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk06), fil06, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est99, fil99, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(esttt, filtt, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(estbu, filbu, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    fildd = pic * ".estot.txt" ; line = @sprintf("%f %d %d %f %f %f %d %d %f %f %f %d\n", 0.0, tarit, tarib, avg[PTTT], avg[PTBB], est99[tarit,tarib], finit, finib, avg[PFTT], avg[PFBB], est99[finit,finib], smoopass)
    fpa = ouvre(fildd, "w")  ; write(fpa, line)
    write(fpa, "Cov(B,T)\n") ; write(fpa, "Cov(B,D)\n") ; write(fpa, "Cov(B,V)\n")
    write(fpa, "Cov(D,T)\n") ; write(fpa, "Cov(T,V)\n") ; write(fpa, "Cov(D,V)\n")
    close(fpa)
    if PLOTPROG
      print("grads --quiet -blc \"ErrorsInVariables.plot.solution $fil00 $fil01 $fil02 $fil03 $fil04 $fil05 $fil06 $fil99 $fildd $filtt $filbu $pic\"\n")
        run(`grads --quiet -blc  "ErrorsInVariables.plot.solution $fil00 $fil01 $fil02 $fil03 $fil04 $fil05 $fil06 $fil99 $fildd $filtt $filbu $pic"`)
    end
    !keepnc && (rm(fil00) ; rm(fil01) ; rm(fil02) ; rm(fil03) ; rm(fil04) ; rm(fil05) ; rm(fil06) ; rm(fil99) ; rm(filtt) ; rm(filbu) ; rm(fildd))
  end

  if echotxt != [] && solve != false
    @printf("\nnumber of collocations including outliers = %15d\n", length(cc))
    @printf(  "number of collocations excluding outliers = %15d\n", length(cc[mask]))
    for a in echotxt
      @printf("\nrcm[%d,CTRU,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CTRU,MOLR], rcm[a,CTRU,MEIV], rcm[a,CTRU,MRLR])
      @printf(  "rcm[%d,CALP,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CALP,MOLR], rcm[a,CALP,MEIV], rcm[a,CALP,MRLR])
      @printf(  "rcm[%d,CBET,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CBET,MOLR], rcm[a,CBET,MEIV], rcm[a,CBET,MRLR])
      @printf(  "rcm[%d,CLAM,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CLAM,MOLR], rcm[a,CLAM,MEIV], rcm[a,CLAM,MRLR])
      @printf(  "rcm[%d,CERI,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CERI,MOLR], rcm[a,CERI,MEIV], rcm[a,CERI,MRLR])
      @printf(  "rcm[%d,CERT,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CERT,MOLR], rcm[a,CERT,MEIV], rcm[a,CERT,MRLR])
    end
    @printf("\navg[RRCC]           = %15.8f\n",               avg[RRCC])
    @printf(  "avg[RRUU]           = %15.8f\n",               avg[RRUU])
    @printf(  "avg[PDIS]           = %15.8f\n\n",             avg[PDIS])
    @printf("           Alpha            Beta    VAR(I) Linear       Nonlinear    Unassociated   VAR(N) Linear       Nonlinear    Unassociated\n")
    @printf(" %15.8f %15.8f %s\n", precalalp, precalbet, sumlin)
  end
  solve == false && print("\ncpseiv ERROR : returning a missing solution\n\n")
  solve == false && (rcm[:,:,MEIV] .= missval ; sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", missval, missval, missval, missval, missval, missval))
  return(avg, rcm, sumlin)
end

function cpseiv(aa::Array{Float64,1}, bb::Array{Float64,1}, cc::Array{Float64,1}, dd::Array{Float64,1}, ee::Array{Float64,1}, precalalp::Float64, precalbet::Float64, uu::Array{Float64,1}; missval = MISS, detmcd = true, limmcd = MCDTRIM, pic = "", picrng = 1.0:0.0, keepnc = false, echotxt = [])
  avg    = fill(missval, PDIS)
  rcm    = fill(missval, (RRCM, CRCM, MRCM))
  sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", missval, missval, missval, missval, missval, missval)
  length(cc) < 50 && return(avg, rcm, sumlin)

# if detmcd                                                                   # get a mask to exclude outliers using DetMCD in R
#   temp = [aa bb cc dd ee uu]                                                # (before precalibration, avgc = remp[:center][3],
#   remp = rcopy(R"DetMCD($temp, alpha = $limmcd)")                           # varc = remp[:cov][3,3], cvcu = remp[:cov][3,6])
#   mask = falses(length(cc)) ; for a in remp[:Hsubsets]  mask[a] = true  end
# else
    mask =  trues(length(cc))
# end

  if precalalp == MISS && precalbet == MISS
    avgc = mean(cc[mask]) ; varc = var(cc[mask])
    avgu = mean(uu[mask]) ; varu = var(uu[mask])
    cvcu = cov(cc[mask], uu[mask])
    precalbet = sign(cvcu) * (varu / varc)^0.5
    precalalp = avgu - precalbet * avgc
    @printf("\npreliminary calibration by variance matchng alp = %15.8f bet = %15.8f\n", precalalp, precalbet)
  else
    @printf("\npreliminary calibration by specified values alp = %15.8f bet = %15.8f\n", precalalp, precalbet)
  end
  for a = 1:length(uu)
    uu[a] -= precalalp ; uu[a] /= precalbet
  end

  avga = mean(aa[mask]) ; vara = var(aa[mask])
  avgb = mean(bb[mask]) ; varb = var(bb[mask])
  avgc = mean(cc[mask]) ; varc = var(cc[mask])
  avgd = mean(dd[mask]) ; vard = var(dd[mask])
  avge = mean(ee[mask]) ; vare = var(ee[mask])
  avgu = mean(uu[mask]) ; varu = var(uu[mask])
  cvau = cov(aa[mask], uu[mask])
  cvbu = cov(bb[mask], uu[mask])
  cvcu = cov(cc[mask], uu[mask])
  cvdu = cov(dd[mask], uu[mask])
  cveu = cov(ee[mask], uu[mask])
  cvab = cov(aa[mask], bb[mask])
  cvac = cov(aa[mask], cc[mask])
  cvad = cov(aa[mask], dd[mask])
  cvae = cov(aa[mask], ee[mask])
  cvbc = cov(bb[mask], cc[mask])
  cvbd = cov(bb[mask], dd[mask])
  cvbe = cov(bb[mask], ee[mask])                                              # get RCM OLR (no C error) and RLR (no U error) values
  cvcd = cov(cc[mask], dd[mask])                                              # for available predictive samples among ABCDE and STUVW
  cvce = cov(cc[mask], ee[mask])                                              # (for simplicity, CLAM = 0 for OLR-STVW and RLR-ABDE)
  cvde = cov(dd[mask], ee[mask])
  avg = [avga, avgb, avgc, avgd, avge, MISS, MISS, avgu, MISS, MISS, length(cc[mask]), precalalp, precalbet, MISS, MISS, MISS, MISS, MISS]

  rcm = zeros(RRCM, CRCM, MRCM)
  a = RRAA ; rcm[a,CCVA,:] .= vara ; rcm[a,CCVB,:] .= cvab ; rcm[a,CCVC,:] .= cvac ; rcm[a,CCVD,:] .= cvad ; rcm[a,CCVE,:] .= cvae ; rcm[a,CCVU,:] .= cvau
  a = RRBB ; rcm[a,CCVA,:] .= cvab ; rcm[a,CCVB,:] .= varb ; rcm[a,CCVC,:] .= cvbc ; rcm[a,CCVD,:] .= cvbd ; rcm[a,CCVE,:] .= cvbe ; rcm[a,CCVU,:] .= cvbu
  a = RRCC ; rcm[a,CCVA,:] .= cvac ; rcm[a,CCVB,:] .= cvbc ; rcm[a,CCVC,:] .= varc ; rcm[a,CCVD,:] .= cvcd ; rcm[a,CCVE,:] .= cvce ; rcm[a,CCVU,:] .= cvcu
  a = RRDD ; rcm[a,CCVA,:] .= cvad ; rcm[a,CCVB,:] .= cvbd ; rcm[a,CCVC,:] .= cvcd ; rcm[a,CCVD,:] .= vard ; rcm[a,CCVE,:] .= cvde ; rcm[a,CCVU,:] .= cvdu
  a = RREE ; rcm[a,CCVA,:] .= cvae ; rcm[a,CCVB,:] .= cvbe ; rcm[a,CCVC,:] .= cvce ; rcm[a,CCVD,:] .= cvde ; rcm[a,CCVE,:] .= vare ; rcm[a,CCVU,:] .= cveu
  a = RRUU ; rcm[a,CCVA,:] .= cvau ; rcm[a,CCVB,:] .= cvbu ; rcm[a,CCVC,:] .= cvcu ; rcm[a,CCVD,:] .= cvdu ; rcm[a,CCVE,:] .= cveu ; rcm[a,CCVU,:] .= varu
  c = MOLR
  a = RRAA ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvac / varc ; rcm[a,CALP,c] = avga - rcm[a,CBET,c] * avgc
  a = RRBB ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvbc / varc ; rcm[a,CALP,c] = avgb - rcm[a,CBET,c] * avgc
  a = RRCC ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] =         1.0 ; rcm[a,CALP,c] =                         0.0
  a = RRDD ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcd / varc ; rcm[a,CALP,c] = avgd - rcm[a,CBET,c] * avgc
  a = RREE ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvce / varc ; rcm[a,CALP,c] = avge - rcm[a,CBET,c] * avgc
  a = RRUU ; rcm[a,CTRU,c] = varc ; rcm[a,CBET,c] = cvcu / varc ; rcm[a,CALP,c] = avgu - rcm[a,CBET,c] * avgc
  a = RRAA ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vara - rcm[a,CBET,c] * cvac
  a = RRBB ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varb - rcm[a,CBET,c] * cvbc
  a = RRCC ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] =                         0.0
  a = RRDD ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vard - rcm[a,CBET,c] * cvcd
  a = RREE ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vare - rcm[a,CBET,c] * cvce
  a = RRUU ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varu - rcm[a,CBET,c] * cvcu
  a = RRCC ; rcm[a,CLIN,c] =                                          100
             rcm[a,CNOL,c] =                                          0.0
             rcm[a,CNOA,c] =                                          0.0
  a = RRUU ; rcm[a,CLIN,c] = rcm[a,CBET,c]^2 * rcm[a,CTRU,c] / varu * 100
             rcm[a,CNOL,c] =                                          0.0
             rcm[a,CNOA,c] =                   rcm[a,CERI,c] / varu * 100
  c = MRLR
  a = RRAA ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvau / cvcu ; rcm[a,CALP,c] = avga - rcm[a,CBET,c] * avgc
  a = RRBB ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvbu / cvcu ; rcm[a,CALP,c] = avgb - rcm[a,CBET,c] * avgc
  a = RRCC ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] =         1.0 ; rcm[a,CALP,c] =                         0.0
  a = RRDD ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cvdu / cvcu ; rcm[a,CALP,c] = avgd - rcm[a,CBET,c] * avgc
  a = RREE ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = cveu / cvcu ; rcm[a,CALP,c] = avge - rcm[a,CBET,c] * avgc
  a = RRUU ; rcm[a,CTRU,c] = cvcu * cvcu / varu ; rcm[a,CBET,c] = varu / cvcu ; rcm[a,CALP,c] = avgu - rcm[a,CBET,c] * avgc
  a = RRAA ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vara - cvau * cvau / varu
  a = RRBB ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varb - cvbu * cvbu / varu
  a = RRCC ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = varc - cvcu * cvcu / varu
  a = RRDD ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vard - cvdu * cvdu / varu
  a = RREE ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] = vare - cveu * cveu / varu
  a = RRUU ; rcm[a,CLAM,c] = 0.0 ; rcm[a,CERI,c] = rcm[a,CERT,c] =                       0.0
  a = RRCC ; rcm[a,CLIN,c] = rcm[a,CTRU,c] / varc * 100
             rcm[a,CNOL,c] =                        0.0
             rcm[a,CNOA,c] = rcm[a,CERI,c] / varc * 100
  a = RRUU ; rcm[a,CLIN,c] =                        100
             rcm[a,CNOL,c] =                        0.0
             rcm[a,CNOA,c] =                        0.0

  function weak(tttt::Float64, betu::Float64)                                 # provide weak solution constraints
    mskid = 0                                                                 # and a negative covariance mask
    ccBT = varc -          tttt                        ; ccBT <= 0 && (mskid += 1)
    cuBT = cvcu - betu   * tttt                        ; cuBT <= 0 && (mskid += 10)
    uuBT = varu - betu^2 * tttt                        ; uuBT <= 0 && (mskid += 100)
    lama = (betu * cvac - cvau) / (betu * cvbc - cvbu) ; lama <  0 && (mskid += 2)    #; lama > 1 && (mskid += 4)
    lamb = (betu * cvbc - cvbu) / (betu * ccBT - cuBT) ; lamb <  0 && (mskid += 20)   #; lamb > 1 && (mskid += 40)
    lamd = (betu * cvcd - cvdu) / (betu * ccBT - cuBT) ; lamd <  0 && (mskid += 200)  #; lamd > 1 && (mskid += 400)
    lame = (betu * cvce - cveu) / (betu * cvcd - cvdu) ; lame <  0 && (mskid += 2000) #; lame > 1 && (mskid += 4000)
    beta = (cvac - lama * lamb * cuBT) / tttt
    betb = (cvbc -        lamb * cuBT) / tttt
    betd = (cvcd - lamd *        cuBT) / tttt
    bete = (cvce - lamd * lame * cuBT) / tttt
    aaBT = vara -        beta^2 * tttt                 ; aaBT <= 0 && (mskid += 10000)
    bbBT = varb -        betb^2 * tttt                 ; bbBT <= 0 && (mskid += 100000)
    ddBT = vard -        betd^2 * tttt                 ; ddBT <= 0 && (mskid += 1000000)
    eeBT = vare -        bete^2 * tttt                 ; eeBT <= 0 && (mskid += 10000000)
#   auBT = cvau - beta * betu   * tttt                 ; auBT <= 0 && (mskid += 20000)
#   buBT = cvbu - betb * betu   * tttt                 ; buBT <= 0 && (mskid += 200000)
#   duBT = cvdu - betd * betu   * tttt                 ; duBT <= 0 && (mskid += 2000000)
#   euBT = cveu - bete * betu   * tttt                 ; euBT <= 0 && (mskid += 20000000)
    eeaa = aaBT - lama^2 * bbBT                        ; eeaa <= 0 && (mskid += 100000000)
    eebb = bbBT - lamb^2 * ccBT                        ; eebb <= 0 && (mskid += 1000000000)
    eecc = ccBT -          cuBT                        ; eecc <= 0 && (mskid += 10000000000)
    eedd = ddBT - lamd^2 * ccBT                        ; eedd <= 0 && (mskid += 100000000000)
    eeee = eeBT - lame^2 * ddBT                        ; eeee <= 0 && (mskid += 1000000000000)
    eeuu = uuBT -          cuBT                        ; eeuu <= 0 && (mskid += 10000000000000)

    wkab = abs(cvab - beta * betb * tttt - lama *                      bbBT)  # (weak constraints are just the covariance
    wkbd = abs(cvbd - betb * betd * tttt -        lamb * lamd *        ccBT)  # eqns that exclude those involving C and U)
    wkde = abs(cvde - betd * bete * tttt -                      lame * ddBT)
    wkad = abs(cvad - beta * betd * tttt - lama * lamb * lamd *        ccBT)
    wkae = abs(cvae - beta * bete * tttt - lama * lamb * lamd * lame * ccBT)
    wkbe = abs(cvbe - betb * bete * tttt -        lamb * lamd * lame * ccBT)
    wkto =           (wkab +     wkbd +     wkde +     wkad +     wkae +     wkbe) / 6
    return(mskid, log(wkab), log(wkbd), log(wkde), log(wkad), log(wkae), log(wkbe), log(wkto))
  end

  solve = true                                                                # search for positive-variance EIV solutions
  mintt =  0.0                                                                # that are bounded by OLR and RLR and a wide
  maxtt =  2.0 * varc                                                         # range of shared true variance (tttt) values
  minbu = cvcu / varc                                                         # (and allow that there might be no solution)
  maxbu = varu / cvcu
  minbu > maxbu && ((minbu, maxbu) = (maxbu, minbu))
  rngtt = collect(range(mintt, stop = maxtt, length = ESTIM + 2))[2:end-1]
  rngbu = collect(range(minbu, stop = maxbu, length = ESTIM + 2))[2:end-1]
  est00 = Array{Float64}(undef, ESTIM, ESTIM)
  est01 = Array{Float64}(undef, ESTIM, ESTIM)
  est02 = Array{Float64}(undef, ESTIM, ESTIM)
  est03 = Array{Float64}(undef, ESTIM, ESTIM)
  est04 = Array{Float64}(undef, ESTIM, ESTIM)
  est05 = Array{Float64}(undef, ESTIM, ESTIM)
  est06 = Array{Float64}(undef, ESTIM, ESTIM)
  est99 = Array{Float64}(undef, ESTIM, ESTIM)
  for (a, vala) in enumerate(rngtt), (b, valb) in enumerate(rngbu)
    est00[a,b], est01[a,b], est02[a,b], est03[a,b], est04[a,b], est05[a,b], est06[a,b], est99[a,b] = weak(vala, valb)
  end
  !any(x -> x == 0, est00) && (solve = false)                                 # find a positive-variance consensus-path
  !solve && print("\ncpseiv ERROR : no positive variance solution\n\n")       # solution by covariances that exclude CU

  solve, tarit, tarib, finit, finib, esttt, estbu, smoopass, msk01, msk02, msk03, msk04, msk05, msk06 = consensus(solve, rngtt, rngbu, est00, est01, est02, est03, est04, est05, est06)
  tttt = avg[PFTT] = rngtt[finit] ; betu = avg[PFBB] = rngbu[finib]
         avg[PTTT] = rngtt[tarit] ;        avg[PTBB] = rngbu[tarib]
         avg[PDIS] = (((finit - tarit)^2 + (finib - tarib)^2) / (2 * ESTIM^2))^0.5

  ccBT = varc -          tttt                                                 # derive the EIV metrics and complete the RCM
  cuBT = cvcu - betu   * tttt
  uuBT = varu - betu^2 * tttt
  lama = (betu * cvac - cvau) / (betu * cvbc - cvbu) ; beta = (cvac - lama * lamb * cuBT) / tttt
  lamb = (betu * cvbc - cvbu) / (betu * ccBT - cuBT) ; betb = (cvbc -        lamb * cuBT) / tttt
  lamd = (betu * cvcd - cvdu) / (betu * ccBT - cuBT) ; betd = (cvcd - lamd *        cuBT) / tttt
  lame = (betu * cvce - cveu) / (betu * cvcd - cvdu) ; bete = (cvce - lamd * lame * cuBT) / tttt
  aaBT = vara -        beta^2 * tttt
  bbBT = varb -        betb^2 * tttt
  ddBT = vard -        betd^2 * tttt
  eeBT = vare -        bete^2 * tttt
  eeaa = aaBT - lama^2 * bbBT
  eebb = bbBT - lamb^2 * ccBT
  eecc = ccBT -          cuBT
  eedd = ddBT - lamd^2 * ccBT
  eeee = eeBT - lame^2 * ddBT
  eeuu = uuBT -          cuBT

  c = MEIV
  a = RRAA ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = beta ; rcm[a,CALP,c] = avga - beta * avgc
  a = RRBB ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betb ; rcm[a,CALP,c] = avgb - betb * avgc
  a = RRCC ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] =  1.0 ; rcm[a,CALP,c] =                0.0
  a = RRDD ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betd ; rcm[a,CALP,c] = avgd - betd * avgc
  a = RREE ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = bete ; rcm[a,CALP,c] = avge - bete * avgc
  a = RRUU ; rcm[a,CTRU,c] = tttt ; rcm[a,CBET,c] = betu ; rcm[a,CALP,c] = avgu - betu * avgc
  a = RRAA ; rcm[a,CLAM,c] = lama ; rcm[a,CERI,c] = eeaa ; rcm[a,CERT,c] = aaBT
  a = RRBB ; rcm[a,CLAM,c] = lamb ; rcm[a,CERI,c] = eebb ; rcm[a,CERT,c] = bbBT
  a = RRCC ; rcm[a,CLAM,c] =  0.0 ; rcm[a,CERI,c] = eecc ; rcm[a,CERT,c] = ccBT
  a = RRDD ; rcm[a,CLAM,c] = lamd ; rcm[a,CERI,c] = eedd ; rcm[a,CERT,c] = ddBT
  a = RREE ; rcm[a,CLAM,c] = lame ; rcm[a,CERI,c] = eeee ; rcm[a,CERT,c] = eeBT
  a = RRUU ; rcm[a,CLAM,c] =  0.0 ; rcm[a,CERI,c] = eeuu ; rcm[a,CERT,c] = uuBT
  a = RRCC ; rcm[a,CLIN,c] =          tttt / varc * 100 ; rcm[a,CNOL,c] = cuBT / varc * 100 ; rcm[a,CNOA,c] = eecc / varc * 100
  a = RRUU ; rcm[a,CLIN,c] = betu^2 * tttt / varu * 100 ; rcm[a,CNOL,c] = cuBT / varu * 100 ; rcm[a,CNOA,c] = eeuu / varu * 100

  sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", rcm[RRCC,CLIN,c], rcm[RRCC,CNOL,c], rcm[RRCC,CNOA,c],
                                                                    rcm[RRUU,CLIN,c], rcm[RRUU,CNOL,c], rcm[RRUU,CNOA,c])

  if pic != ""                                                                # then plot the EIV solution
    if last(picrng) > first(picrng)
      intpic = collect(picrng)[2:end] .- 0.5 * step(picrng) ; lenpic = length(intpic)
      fil01 = pic * ".hst01.nc" ; isfile(fil01) && rm(fil01) ; nccreer(fil01, 1, [0.0], intpic, missval)
      fil02 = pic * ".hst02.nc" ; isfile(fil02) && rm(fil02) ; nccreer(fil02, 1, [0.0], intpic, missval)
      fil03 = pic * ".hst03.nc" ; isfile(fil03) && rm(fil03) ; nccreer(fil03, 1, [0.0], intpic, missval)
      fil04 = pic * ".hst04.nc" ; isfile(fil04) && rm(fil04) ; nccreer(fil04, 1, [0.0], intpic, missval)
      fil05 = pic * ".hst05.nc" ; isfile(fil05) && rm(fil05) ; nccreer(fil05, 1, [0.0], intpic, missval)
      fil06 = pic * ".hst06.nc" ; isfile(fil06) && rm(fil06) ; nccreer(fil06, 1, [0.0], intpic, missval)
      fil51 = pic * ".hst51.nc" ; isfile(fil51) && rm(fil51) ; nccreer(fil51, 1, [0.0], intpic, missval)
      fil52 = pic * ".hst52.nc" ; isfile(fil52) && rm(fil52) ; nccreer(fil52, 1, [0.0], intpic, missval)
      fil53 = pic * ".hst53.nc" ; isfile(fil53) && rm(fil53) ; nccreer(fil53, 1, [0.0], intpic, missval)
      fil54 = pic * ".hst54.nc" ; isfile(fil54) && rm(fil54) ; nccreer(fil54, 1, [0.0], intpic, missval)
      fil55 = pic * ".hst55.nc" ; isfile(fil55) && rm(fil55) ; nccreer(fil55, 1, [0.0], intpic, missval)
      fil56 = pic * ".hst56.nc" ; isfile(fil56) && rm(fil56) ; nccreer(fil56, 1, [0.0], intpic, missval)
      ncwrite(logfithist(aa[  mask], picrng), fil01, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(bb[  mask], picrng), fil02, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(cc[  mask], picrng), fil03, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(dd[  mask], picrng), fil04, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(ee[  mask], picrng), fil05, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(uu[  mask], picrng), fil06, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(aa[.!mask], picrng), fil51, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(bb[.!mask], picrng), fil52, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(cc[.!mask], picrng), fil53, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(dd[.!mask], picrng), fil54, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(ee[.!mask], picrng), fil55, "tmp", start=[1,1,1], count=[lenpic,1,1])
      ncwrite(logfithist(uu[.!mask], picrng), fil56, "tmp", start=[1,1,1], count=[lenpic,1,1])
      filaa = pic * ".txt" ; lena = length(cc[mask]) ; lenb = length(cc[.!mask]) ; lenc = lena + lenb
      line = @sprintf("%d %d %7.1f %7.1f %9.5f %.0f %7.2f\n", lena, lenb, 100 * lenb / lenc, 100 * limmcd, ecdfsup(cc[mask], uu[mask]), difhist(cc[mask], uu[mask], picrng), precalbet)
      fpa = ouvre(filaa, "w")          ; write(fpa,     line)
      write(fpa, statline(cc[  mask])) ; write(fpa, statline(uu[  mask])) ; write(fpa, statline(aa[  mask]))
      write(fpa, statline(bb[  mask])) ; write(fpa, statline(dd[  mask])) ; write(fpa, statline(ee[  mask]))
      write(fpa, statline(cc[.!mask])) ; write(fpa, statline(uu[.!mask])) ; write(fpa, statline(aa[.!mask]))
      write(fpa, statline(bb[.!mask])) ; write(fpa, statline(dd[.!mask])) ; write(fpa, statline(ee[.!mask]))
      write(fpa, "Calibrated\n")       ; write(fpa, "Uncalibrat\n")       ; write(fpa, "Calib (T-2)\n")
      write(fpa, "Calib (T-1)\n")      ; write(fpa, "Calib (T+1)\n")      ; write(fpa, "Calib (T+2)\n")
      close(fpa)
      if PLOTPROG
        print("grads --quiet -blc \"ErrorsInVariables.plot.distribution $fil03 $fil06 $fil01 $fil02 $fil04 $fil05 $fil53 $fil56 $fil51 $fil52 $fil54 $fil55 $filaa $pic.dist\"\n")
          run(`grads --quiet -blc  "ErrorsInVariables.plot.distribution $fil03 $fil06 $fil01 $fil02 $fil04 $fil05 $fil53 $fil56 $fil51 $fil52 $fil54 $fil55 $filaa $pic.dist"`)
      end
      !keepnc && (rm(fil01) ; rm(fil02) ; rm(fil03) ; rm(fil04) ; rm(fil05) ; rm(fil06) ; rm(fil51) ; rm(fil52) ; rm(fil53) ; rm(fil54) ; rm(fil55) ; rm(fil56) ; rm(filaa))

      filbb = pic * ".hstin.nc" ; isfile(filbb) && rm(filbb) ; nccreer(filbb, 1, intpic, intpic, missval)
      ncwrite(logfitdoub(cc[mask], uu[mask], picrng), filbb, "tmp", start=[1,1,1], count=[lenpic,lenpic,1])
      filcc = pic * ".txt" ; rcmsave(avg, rcm, filcc; sumlin = sumlin)
      if PLOTPROG
        print("grads --quiet -blc \"ErrorsInVariables.plot.doublebution $filbb $filcc $pic.doub\"\n")
          run(`grads --quiet -blc  "ErrorsInVariables.plot.doublebution $filbb $filcc $pic.doub"`)
      end
      !keepnc && (rm(filbb) ; rm(filcc * ".IAVG") ; rm(filcc * ".MOLR") ; rm(filcc * ".MEIV") ; rm(filcc * ".MRLR"))
    end

    for (a, vala) in enumerate(rngtt), (b, valb) in enumerate(rngbu)
      est01[a,b] > CUTOFF && (est01[a,b] = CUTOFF)
      est02[a,b] > CUTOFF && (est02[a,b] = CUTOFF)
      est03[a,b] > CUTOFF && (est03[a,b] = CUTOFF)
      est04[a,b] > CUTOFF && (est04[a,b] = CUTOFF)
      est05[a,b] > CUTOFF && (est05[a,b] = CUTOFF)
      est06[a,b] > CUTOFF && (est06[a,b] = CUTOFF)
    end
    fil00 = pic * ".est00.nc" ; isfile(fil00) && rm(fil00) ; nccreer(fil00, 1, rngbu, rngtt, missval)
    fil01 = pic * ".est01.nc" ; isfile(fil01) && rm(fil01) ; nccreer(fil01, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil02 = pic * ".est02.nc" ; isfile(fil02) && rm(fil02) ; nccreer(fil02, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil03 = pic * ".est03.nc" ; isfile(fil03) && rm(fil03) ; nccreer(fil03, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil04 = pic * ".est04.nc" ; isfile(fil04) && rm(fil04) ; nccreer(fil04, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil05 = pic * ".est05.nc" ; isfile(fil05) && rm(fil05) ; nccreer(fil05, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil06 = pic * ".est06.nc" ; isfile(fil06) && rm(fil06) ; nccreer(fil06, 1, rngbu, rngtt, missval; vnames = ["tmp", "msk"])
    fil99 = pic * ".est99.nc" ; isfile(fil99) && rm(fil99) ; nccreer(fil99, 1, rngbu, rngtt, missval)
    filtt = pic * ".esttt.nc" ; isfile(filtt) && rm(filtt) ; nccreer(filtt, 1, rngbu, rngtt, missval)
    filbu = pic * ".estbu.nc" ; isfile(filbu) && rm(filbu) ; nccreer(filbu, 1, rngbu, rngtt, missval)
            est00[finit,finib] = -1.0
    ncwrite(est00, fil00, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est01, fil01, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk01), fil01, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est02, fil02, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk02), fil02, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est03, fil03, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk03), fil03, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est04, fil04, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk04), fil04, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est05, fil05, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk05), fil05, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est06, fil06, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1]) ; ncwrite(float(msk06), fil06, "msk", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(est99, fil99, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(esttt, filtt, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    ncwrite(estbu, filbu, "tmp", start=[1,1,1], count=[ESTIM,ESTIM,1])
    fildd = pic * ".estot.txt" ; line = @sprintf("%f %d %d %f %f %f %d %d %f %f %f %d\n", 0.0, tarit, tarib, avg[PTTT], avg[PTBB], est99[tarit,tarib], finit, finib, avg[PFTT], avg[PFBB], est99[finit,finib], smoopass)
    fpa = ouvre(fildd, "w")  ; write(fpa, line)
    write(fpa, "Cov(A,B)\n") ; write(fpa, "Cov(B,D)\n") ; write(fpa, "Cov(D,E)\n")
    write(fpa, "Cov(A,D)\n") ; write(fpa, "Cov(A,E)\n") ; write(fpa, "Cov(B,E)\n")
    close(fpa)
    if PLOTPROG
      print("grads --quiet -blc \"ErrorsInVariables.plot.solution $fil00 $fil01 $fil02 $fil03 $fil04 $fil05 $fil06 $fil99 $fildd $filtt $filbu $pic\"\n")
        run(`grads --quiet -blc  "ErrorsInVariables.plot.solution $fil00 $fil01 $fil02 $fil03 $fil04 $fil05 $fil06 $fil99 $fildd $filtt $filbu $pic"`)
    end
    !keepnc && (rm(fil00) ; rm(fil01) ; rm(fil02) ; rm(fil03) ; rm(fil04) ; rm(fil05) ; rm(fil06) ; rm(fil99) ; rm(filtt) ; rm(filbu) ; rm(fildd))
  end

  if echotxt != [] && solve != false
    @printf("\nnumber of collocations including outliers = %15d\n", length(cc))
    @printf(  "number of collocations excluding outliers = %15d\n", length(cc[mask]))
    for a in echotxt
      @printf("\nrcm[%d,CTRU,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CTRU,MOLR], rcm[a,CTRU,MEIV], rcm[a,CTRU,MRLR])
      @printf(  "rcm[%d,CALP,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CALP,MOLR], rcm[a,CALP,MEIV], rcm[a,CALP,MRLR])
      @printf(  "rcm[%d,CBET,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CBET,MOLR], rcm[a,CBET,MEIV], rcm[a,CBET,MRLR])
      @printf(  "rcm[%d,CLAM,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CLAM,MOLR], rcm[a,CLAM,MEIV], rcm[a,CLAM,MRLR])
      @printf(  "rcm[%d,CERI,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CERI,MOLR], rcm[a,CERI,MEIV], rcm[a,CERI,MRLR])
      @printf(  "rcm[%d,CERT,MOLR/MEIV/MRLR] = %15.8f %15.8f %15.8f\n", a, rcm[a,CERT,MOLR], rcm[a,CERT,MEIV], rcm[a,CERT,MRLR])
    end
    @printf("\navg[RRCC]           = %15.8f\n",               avg[RRCC])
    @printf(  "avg[RRUU]           = %15.8f\n",               avg[RRUU])
    @printf(  "avg[PDIS]           = %15.8f\n\n",             avg[PDIS])
    @printf("           Alpha            Beta    VAR(I) Linear       Nonlinear    Unassociated   VAR(N) Linear       Nonlinear    Unassociated\n")
    @printf(" %15.8f %15.8f %s\n", precalalp, precalbet, sumlin)
  end
  solve == false && print("\ncpseiv ERROR : returning a missing solution\n\n")
  solve == false && (rcm[:,:,MEIV] .= missval ; sumlin = @sprintf(" %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n", missval, missval, missval, missval, missval, missval))
  return(avg, rcm, sumlin)
end

function rcmsave(avg::Array{Float64,1}, rcm::Array{Float64,3}, stump::AbstractString, echo::Bool = true; sumlin = "")
  if stump != ""
    for (c, tail) in enumerate([".MOLR", ".MEIV", ".MRLR"])
      fpa = ouvre(stump * tail, "w", echo)
      for a = 1:RRCM
        for b = 1:CRCM
          tmp = @sprintf(" %15.8f", rcm[a,b,c]) ; write(fpa, tmp)
        end
        tmp   = @sprintf(     "\n"            ) ; write(fpa, tmp)
      end
      close(fpa)
    end

    fpa = ouvre(stump * ".IAVG", "w", echo)
    for a = 1:PDIS
      tmp = @sprintf(" %15.8f", avg[a]) ; write(fpa, tmp)
    end
    tmp   = @sprintf(     "\n"        ) ; write(fpa, tmp)
    sumlin != "" && write(fpa, sumlin)
    close(fpa)
  end
end

function rcmread(stump::AbstractString, echo::Bool = true)
  if stump != ""
    avg = Array{Float64}(undef, PDIS)
    rcm = Array{Float64}(undef, RRCM, CRCM, MRCM)
    for (c, tail) in enumerate([".MOLR", ".MEIV", ".MRLR"])
      fpa = ouvre(stump * tail, "r", echo)
      for a = 1:RRCM
        rcm[a,:,c] = map(x -> parse(Float64, x), split(readline(fpa)))
      end
      close(fpa)
    end

    fpa = ouvre(stump * ".IAVG", "r", echo)
    avg[:]         = map(x -> parse(Float64, x), split(readline(fpa)))
    sumlin         =                                   readline(fpa)
    close(fpa)
  end
  return(avg, rcm, sumlin)
end

function ouvre(fn::AbstractString, mode::AbstractString, echo::Bool = true)
  if mode == "r" && echo  print("reading $fn\n")  end
  if mode == "w" && echo  print("writing $fn\n")  end
  if mode == "a" && echo  print("appding $fn\n")  end

  global fp
  try fp = open(fn, mode) catch ; error("ERROR : ¯\\_(ツ)_/¯ couldn't open $fn\n")  end
  return(fp)
end

function infers(cc::Array{Float64,1}, tail::AbstractString, ss::Array{Float64,1}, tt::Array{Float64,1}, uu::Array{Float64,1}, vv::Array{Float64,1}, ww::Array{Float64,1})
  dirls = readdir()
  stema = @sprintf("spur.%s.spec.reza", tail) ; lena = length(stema)
  map(x -> length(x) > lena && x[1:length(stema)] == stema && rm(x), dirls)

  alph = 0.0 ; beta = 1.0 ; picstem = PLOTCOST ? stema : ""
  (avga, rcma, suma) = cpseiv(cc, alph, beta, ss, tt, uu, vv, ww; pic = picstem, picrng = BANDBIN, keepnc = KEEPNETCDF, echotxt = RRUU)
  rcmsave(avga, rcma, stema; sumlin = suma)
end

function infers(bb::Array{Float64,1}, cc::Array{Float64,1}, dd::Array{Float64,1}, tail::AbstractString, tt::Array{Float64,1}, uu::Array{Float64,1}, vv::Array{Float64,1})
  dirls = readdir()
  stema = @sprintf("spur.%s.spec.reza", tail) ; lena = length(stema)
  map(x -> length(x) > lena && x[1:length(stema)] == stema && rm(x), dirls)

  alph = 0.0 ; beta = 1.0 ; picstem = PLOTCOST ? stema : ""
  (avga, rcma, suma) = cpseiv(bb, cc, dd, alph, beta, tt, uu, vv; pic = picstem, picrng = BANDBIN, keepnc = KEEPNETCDF, echotxt = RRUU)
  rcmsave(avga, rcma, stema; sumlin = suma)
end

function infers(aa::Array{Float64,1}, bb::Array{Float64,1}, cc::Array{Float64,1}, dd::Array{Float64,1}, ee::Array{Float64,1}, tail::AbstractString, uu::Array{Float64,1})
  dirls = readdir()
  stema = @sprintf("spur.%s.spec.reza", tail) ; lena = length(stema)
  map(x -> length(x) > lena && x[1:length(stema)] == stema && rm(x), dirls)

  alph = 0.0 ; beta = 1.0 ; picstem = PLOTCOST ? stema : ""
  (avga, rcma, suma) = cpseiv(aa, bb, cc, dd, ee, alph, beta, uu; pic = picstem, picrng = BANDBIN, keepnc = KEEPNETCDF, echotxt = RRUU)
  rcmsave(avga, rcma, stema; sumlin = suma)
end

valr = parse(Float64, ARGS[3]) / 1000                                         # get the perturbation weight values
vale = parse(Float64, ARGS[5]) / 1000
valb = parse(Float64,  "0100") / 1000

if SEPBASE                                                                    # add measureable truth and Gaussian perturbations
  tttt = ncread(fila,"ttsb", start=[1,1,1,1], count=[-1,-1,-1,-1])[:]
else
  tttt = ncread(fila,"ttme", start=[1,1,1,1], count=[-1,-1,-1,-1])[:]
end
pppp = ncread(filh, ARGS[4], start=  [1,1,1], count=   [-1,-1,-1])[:]
eeee = ncread(fili, ARGS[6], start=  [1,1,1], count=   [-1,-1,-1])[:] ; eeee /= 12^0.5
bbbb = ncread(filj,     "a", start=  [1,1,1], count=   [-1,-1,-1])[:] ; bbbb /= 12^0.5
scal = (var(tttt) / var(eeee))^0.5
scab = (var(tttt) / var(bbbb))^0.5
cccc = tttt + valb * scab * bbbb
uuuu = tttt + vale * scal * eeee

function spearman(uu::Array{Float64,1}, pp::Array{Float64,1}, ff::Float64)    # return either the original index of uu, the sorted index of pp,
  ff <= 0 && return(collect(1:length(uu)))                                    # or a sorted index for uu + pp (for fractions ff between 0 and 1),
  ff >= 1 && return(sortperm(pp))                                             # where uu is rescaled to the interval [0,1], pp (typically a set
  inda =            sortperm(uu)                                              # of unit-variance Gaussian random samples) is rescaled by a value
  mint = uu[inda[  1]]                                                        # on the interval 10^[-14,16], and the returned index is unsorted
  maxt = uu[inda[end]]                                                        # in the same way that uu is unsorted
  vari = 10^((1 - ff) * -14 + ff * 16)
  vals = map(x -> (x - mint) / (maxt - mint), uu) .+ pp * vari
  inda[sortperm(vals[inda])][sortperm(inda)]
end
iiii = spearman(tttt[cols], pppp, valr)                                       # define Spearman perturbations as the order of ABCDE

aa = cccc[cols.-2*sint][iiii]                                                 # then get the predictive samples and a model solution
bb = cccc[cols.-1*sint][iiii]
cc = cccc[cols        ][iiii]
dd = cccc[cols.+1*sint][iiii]
ee = cccc[cols.+2*sint][iiii]
ss = uuuu[cols.-2*sint]
tt = uuuu[cols.-1*sint]
uu = uuuu[cols        ]
vv = uuuu[cols.+1*sint]
ww = uuuu[cols.+2*sint]

# infers(aa, bb, cc, dd, ee, tsst * "." * ARGS[3] * ARGS[4] * "." * ARGS[5] * ARGS[6] * ".ABCDEU",         uu)
# infers(    bb, cc, dd,     tsst * "." * ARGS[3] * ARGS[4] * "." * ARGS[5] * ARGS[6] * ".BCDTUV",     tt, uu, vv)
  infers(        cc,         tsst * "." * ARGS[3] * ARGS[4] * "." * ARGS[5] * ARGS[6] * ".CSTUVW", ss, tt, uu, vv, ww)

exit(0)
