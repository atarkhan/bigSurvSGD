#' Big survival data analysis using stochastic gradient descent
#'
#' Fits Cox Model via stochastic gradient descent (SGD). This implementation avoids computational 
#' instability of the standard Cox Model when datasets are large. Furthermore, it scales up with 
#' very large datasets that do not fit the memory. It also handles large sparse datasets using the 
#' Proximal stochastic gradient descent algorithm.   
#'
#' @param formula a formula in format of Surv(time=time, status=status)~feature1+feature2+... 
#' describing time-to-event variable, status variable, and features to be 
#' included in model. Default is "Surv(time, status)~." that regresses
#' on all the features included in the dataset. 
#' @param data survival dataset. It can be in form of data.frame or a path to a .csv file if
#' we aim not to read data off the memory. If we aim to read data off the memory, it must be 
#' a path to a .csv data. 
#' @param norm.method normalization method before starting the analysis.
#' "center" only centers the features by subtracting the mean, "scale" only
#' scales the features by dividing features to their standard deviation, "normalization"
#' does both centering and scaling, and "none" does not perform any pre-processing. The default
#' is "normalization".
#' @param opt.method optimization algorithm: "SGD" estimates the coefficients
#' using the standard stochastic gradient descent; "ADAM" estimates the coefficients
#' using ADAM optimizer; "AMSGrad" estimates the coefficients using AMSGrad optimizer.
#' The default is "AMSGrad".
#' @param features.mean mean vector of features used for normalization.
#' The default is NULL where our alorithm calculates it.
#' @param features.sd standard deviation vector of features used for normalization.
#' The default is NULL where our alorithm calculates it.
#' @param beta.init initialization for coefficient. The default is NULL where our algorithm
#' starts with an all-zero vector.
#' @param beta.type type of coefficient to be returned. If specified as "single", the last
#' updated coefficient is returned. If specified as "averaged", the Polyak-Ruppert
#' (i.e., average over iterates) is returned. The default is "averaged".
#' @param lr.const proportional constant for the learning rate. The higher values
#' give faster but noisier estimates and vice versa. The default is 0.12 for
#' "AMSGrad" optimizer.
#' @param lr.tau the power of iteration index in the learning rate. The bigger
#' value represents the faster decay in the lerning rate and vice versa. The
#' default is 0.5.
#' @param strata.size strata size. The default is 20 patients per stratum.
#' @param batch.size batch size. The default is 1 stratum per batch.
#' @param num.epoch Number of epochs for the SGD-based algorithms. The default is 100.
#' @param b1 hyper parameter for "AMSGrad" and "ADAM". The default is 0.9.
#' See \url{https://arxiv.org/abs/1412.6980} for "ADMA" and
#' \url{https://arxiv.org/abs/1904.03590} for "AMSGrad".
#' @param b2 hyper parameter for "AMSGrad" and "ADAM". The default is 0.99.
#' @param eps hyper parameter for "AMSGrad" and "ADAM". The default is 1e-8.
#' @param inference.method method for inference, i.e., constructing confidence
#' interval (CI): "bootstrap" constructs CI usin non-parametric bootstrap;
#' "plugin": constructs CI using asymptotic properties of U-statistics;
#' The default is "plugin" which returns estimates, confidence intervals,
#' test statistics, and p-values.
#' @param num.boot number of boostrap resamples. Default is 1000.
#' @param num.epoch.boot number of epochs for each boorstrap resamples.
#' Default is 100.
#' @param boot.method optimization method for bootstrap. Default is "SGD".
#' @param lr.const.boot proportional constant for the learning rate for bootstrap
#' resamples. Defauls is "0.12"
#' @param lr.tau.boot power of iteration index in the learning rate for bootstrap resamples. 
#' Defauls is "0.5"
#' @param num.sample.strata number of sample strata per observation to estimate
#' standard error using plugin method. Default value is 1000.
#' @param sig.level significance level for constructing (1-sig.level) confidence interval.
#' Default is 0.05.
#' @param beta0 null vector of coefficients for calculating p-value using plugin method.
#' Default is zero vector.
#' @param alpha penalty coeficient between 0 and 1. alpha=0 only considers
#' the ridge penlaty and alpha=1 only considers the lasso penalty. Otherwise,
#' it considers a convex combination of these two penalties. Defualt is NULL, i.e.,
#' no penalty.
#' @param lambda coeficient for the elastic net penalty.
#' There are three possible scenarios: (1) If alpha is defined NULL, no penalty
#' (ridge or lasso) is considered regardless of values of lambda; (2)
#' If alpha is not NULL but lambda is NULL, it first calculates
#' the largest value of lambda (lambda.max) for which all coefficients become zero.
#' Then it considers an exponentially decreasing sequence of lambda starting from
#' lambda.max ges toward lambda.min (lambda.min=0.01*lambda.max if p>n, otherewise
#' lambda.min=0.0001*lambda.max) and return their corresponding coefficients.
#' (3) If a value for lambda is specified, our algorithm returns
#' coefficients for specified pair of (lambda, alpha). The default is NULL.
#' @param nlambda number of elements to be considered for scenario (2) above.
#' Default is 100.
#' @param lambda.scale we scale lambda.max to make sure we start with a lambda
#' for which we get all coefficients equal to 0. Default is 1.
#' @param num.strata.lambda number of sample strata to estimate maximum lambda (lambda.max) 
#' when alpha is not NULL and lambda=NULL (see lambda).
#' @param parallel.flag to specify if we want to use parallel computing for
#' inference. Default is "F", i.e., no parallel computing.
#' @param num.cores number of cores for parallel computing. The default is "NULL"
#' for which if parallel.flag=T, it uses all available cores on your system.
#' @param bigmemory.flag determins if data needs to be read off the memory in case
#' data does not fit memory. Default is F, not to use bigmemoty package.
#' @param num.rows.chunk maximum number of rows per chunk to be read off the memory.
#' This is crucial for the large datasets that do not fit memory. 
#' Use fewer number of rows for the large number of features, especially 
#' if you receive an error due to lack of memory. The default value is 1e6 rows.
#' @param col.names a vector of characters for column names of data.
#' If NULL, the column names of dataset "data" will be selected. The default
#' is NULL (i.e., reads columns of given dataset).
#'  
#' 
#' @return coef: Log of hazards ratio. If no inference is used, it returns a vector for estimated
#' coefficients: If inference is used, it returns a matrix including estimates and
#' confidence intervals of coefficients. In case of penalization, it resturns a
#' matrix with columns corresponding to lambdas.
#' @return coef.exp: Exponentiated version of coef (hazards ratio).
#' @return lambda: Returns lambda(s) used for penalizarion.
#' @return alpha: Returns alpha used for penalizarion.
#' @return features.mean: Returns means of features, if given or calculated
#' @return features.sd: Returns standard deviations of features, if given or calculated.
#'
#'
#'
#' @examples
#' \dontrun{
#' # Simulated survival data
#' data(survData) # a dataset with 1000 observations (rows) and 10 features (columns)
#' resultsBig <- bigSurvSGD(formula=Surv(time, status)~.,data=survData, 
#' parallel.flag=TRUE, num.cores=2)
#' resultsBig
#' } 
#' 
#' 
#' @examples
#' \dontrun{
#' # Simulated survival data to be read off the memory
#' data(survData) # a dataset with 1000 observations (rows) and 10 features (columns)
#' # Save dataset survSGD as bigSurvSGD to be read chunk-by-chunk off the memory 
#' write.csv(survData, file = "/home/arash/bigSurvData.csv", row.names = FALSE) 
#' resultsBigOffMemory <- bigSurvSGD(formula=Surv(time, status)~.,data="/home/arash/bigSurvData.csv", 
#' bigmemory.flag=TRUE, parallel.flag=TRUE, num.cores=2)
#' resultsBigOffMemory
#' }
#' 
#' 
#' @examples
#' \dontrun{
#' # Simulated sparse survival data
#' data(sparseSurvData) # a sparse data with 100 observations (rows) and 150 features (columns)
#' resultsBigSparse <- bigSurvSGD(formula=Surv(time, status)~.,data=sparseSurvData, 
#' alpha=0.9, lambda=0.1)
#' resultsBigSparse
#' }
#' 
#' @export
bigSurvSGD <- function(formula=Surv(time=time, status=status)~., data,
                               norm.method="standardize", features.mean=NULL, features.sd=NULL,
                               opt.method="AMSGrad", beta.init=NULL, beta.type="averaged",
                               lr.const=0.12, lr.tau=0.5, strata.size=20, batch.size=1,
                               num.epoch=100, b1=0.9, b2=0.99, eps=1e-8, inference.method="plugin",
                               num.boot=1000, num.epoch.boot=100, boot.method="SGD", lr.const.boot=0.12,
                               lr.tau.boot=0.5, num.sample.strata=1000, sig.level=0.05, beta0=0,
                               alpha=NULL, lambda=NULL, nlambda=100, num.strata.lambda=10, lambda.scale=1,
                               parallel.flag=F, num.cores=NULL, 
                               bigmemory.flag=F, num.rows.chunk=1e6, col.names=NULL){
  
  
  #######################################################################
  # Reading big data off the memory
  if (!bigmemory.flag){ # if not using bigmemory
    if (!is.data.frame(data)){ # if you give a path for your data
      big.data <- read.csv(file = data)
    }else{ # # if you give data frame
      big.data <- data
    }
  } else { # if you aim to use bigmemory
    # If you provide an address to read your data, it will be read as bigmemmory
    if (!is.null(col.names)){ # if dataset does not feature names and you provide it as "col.names"
      big.data <- read.big.matrix(filename=data, sep=",",
                                  skip=0, header = T, col.names = col.names)
    } else { # if dataset already has feature names
      big.data <- read.big.matrix(filename=data, sep=",",
                                  skip=0, header = T)
    }
  }
  # Number of rows in the original
  num.rows.big <- nrow(big.data)
  #######################################################################
  
  
  #######################################################################
  # Check if there is at least one covariate
  if (ncol(big.data) < 3){
    stop("data must have 3 or more columns: time, status, and at least one feature")
  }
  # Choose a smaller number of strata size if number of rows are less than strata size times mini-batch size
  if (nrow(big.data) < 2){
    stop("Sample size is too small (with size less than 2)")
  }
  #######################################################################
  
  
  #######################################################################
  ## All variables in your formula
  all.variables <- all.vars(formula)
  # indices for "time" and "status" (the first two elements of )
  surv.indices <- match(all.variables[1:2], colnames(big.data))
  if (length(all.variables)==3 & all.variables[3]=="."){ # if all features must be used in the analysis
    # indices for desired features except "time" and "status"
    features.indices <- setdiff(1:NCOL(big.data), surv.indices)
    sub.col.names <- colnames(big.data)[features.indices]
  }else{ # If a subset of feature(s) must be used in the analysis
    # indices for desired features except "time" and "status"
    features.indices <- match(all.variables[3:length(all.variables)], colnames(big.data))
    sub.col.names <- all.variables[3:length(all.variables)]
  }
  #######################################################################
  
  
  #######################################################################
  ## check if strata and mini-batch sizes are small enough for given number observations
  ## If not, decrease them until at least we have two mini-batches of strata
  
  chengeStrataBatch <- (strata.size>floor(num.rows.big/batch.size)) & (strata.size>2)
  while((strata.size>floor(num.rows.big/batch.size)) & (strata.size>2)){
    if (batch.size>1){
      batch.size <- max(floor(batch.size/2),1)
    }else{
      strata.size <- max(floor(num.rows.big/batch.size),2)
    }
  }
  if(chengeStrataBatch){
    warning(paste0("Strata size times batch size is greater than number of observations.\n This package resizes them to strata size = ", strata.size, " and batch size = ", batch.size))
  }
  #######################################################################
  
  
  
  #######################################################################
  ## Estimate mean and standard deviation of features for normalization
  # Number of chunks must be read in case there are many subjects
  num.sub.sample <- floor(num.rows.big/num.rows.chunk)
  
  chunks.length <- c(0, rep(num.rows.chunk, floor(num.rows.big/num.rows.chunk)),
                     if(num.rows.big %% num.rows.chunk != 0){num.rows.big %% num.rows.chunk})
  
  if (is.null(features.mean) & is.null(features.sd)){ # if mean and standard deviation of features were not provided
    if (norm.method == "center"){ # if only centering is needed
      n2 <- 0
      features.mean <- 0
      for (i in 1:(length(chunks.length)-1)){
        indices.chunk <- (sum(chunks.length[1:i])+1):(sum(chunks.length[1:(i+1)]))
        sub.data <- big.data[indices.chunk, features.indices]
        n1 <- NROW(sub.data)
        if (NCOL(sub.data) > 1){
          M1 <- colMeans(sub.data, na.rm = T)
        }else{
          M1 <- mean(sub.data, na.rm = T)
        }
        features.mean <- (n1*M1+n2*features.mean)/(n1+n2)
        n2 <- n1+n2
      }
      features.sd <- rep(1,NCOL(sub.data))
    }else if (norm.method == "scale" || norm.method == "standardize") { # If scaling or normalization is needed
      n2 <- 0
      features.mean <- 0
      features.sd <- 0
      for (i in 1:(length(chunks.length)-1)){
        indices.chunk <- (sum(chunks.length[1:i])+1):(sum(chunks.length[1:(i+1)]))
        sub.data <- big.data[indices.chunk, features.indices]
        n1 <- NROW(sub.data)
        if (NCOL(sub.data) > 1){
          M1 <- colMeans(sub.data, na.rm = T)
        }else{
          M1 <- mean(sub.data, na.rm = T)
        }
        if (NCOL(sub.data) > 1){
          S1 <- colMeans(sub.data^2, na.rm = T)-M1^2
        }else{
          S1 <- mean(sub.data^2, na.rm = T)-M1^2
        }
        M2 <- features.mean
        S2 <- features.sd
        features.mean <- (n1*M1+n2*M2)/(n1+n2)
        features.sd <- 1/(n1+n2)*(n1*S1+n2*S2+(n1*n2)/(n1+n2)*(M1-M2)^2)
        n2 <- n1+n2
      }
      features.sd <- sqrt(features.sd)
    } else {
      features.mean <- rep(0,NCOL(sub.data))
      features.sd <- rep(1,NCOL(sub.data))
    }
  }
  
  # Check if there is at least one covariate
  if (sum(features.sd==0) > 0){
    stop(paste0("feature(s) ", colnames(big.data)[features.indices][which(features.sd==0)],
                " is/are constant without any variability"))
  }
  #######################################################################
  
  
  
  #######################################################################
  # Maximum penalty coefficient (lambda) for which all coefficients become 0
  lambda.max <- function(big.data, strata.size, num.rows.big, num.rows.chunk, num.strata.lambda,
                         features.indices, surv.indices){
    # maps chunk size to be divisible by strata size
    num.rows.chunk <- strata.size*floor(num.rows.big/strata.size)
    
    # determine how many iterations of data needed to achieve total sample strata
    num.round <- num.strata.lambda/floor(num.rows.big/strata.size)
    num.round.vec <- c(rep(floor(num.rows.big/strata.size),floor(num.round)),
                       if((num.round %% 1)>0){ceiling((num.round %% 1)*floor(num.rows.big/strata.size))})
    
    indices.stratas <- NULL
    for (i in 1:length(num.round.vec)){
      indices.stratas <- c(indices.stratas, sample(1:num.rows.big, num.round.vec[i]*strata.size, replace = F))
    }
    
    # Initialize sum of gradient
    gt.sum <- 0
    
    # determine number of chunks
    num.round <- length(indices.stratas)/num.rows.chunk
    num.round.vec <- cumsum(c(0, rep(num.rows.chunk, floor(num.round)),
                              if((num.round %% 1)>0){ceiling((num.round %% 1)*num.rows.chunk)}))
    # a loop over chunks of data
    for (i.chunk in 1:(length(num.round.vec)-1)){
      # Indices for a chunk of data
      indices.chosen <- indices.stratas[(num.round.vec[i.chunk]+1):(num.round.vec[i.chunk+1])]
      sub.data <- as.matrix(big.data[indices.chosen, c(surv.indices, features.indices)])
      gt.sum <- gt.sum + lambdaMaxC(sub.data, strata.size, norm.method, features.mean, features.sd)
    }
    # returns maximum element of absolute sum of gradients
    return(max(abs(gt.sum))/num.strata.lambda)
  }
  #######################################################################
  
  
  
  #######################################################################
  ## A function that performs a round of estimate with prespecified number of epochs
  ## This function is also used to get estimates from bootstrap resamples for constructing CI
  do.one <- function(big.data=big.data, num.rows.chunk, num.rows.big, norm.method, opt.method,
                     beta.init, beta.type,
                     lr.const, lr.tau, strata.size, batch.size,
                     num.epoch, b1, b2, eps,
                     lambda, alpha, bootstrap,
                     features.indices, surv.indices,
                     features.mean, features.sd){
    
    # initializes beta with
    beta.norm <- beta.init
    
    # Initializes parameters for ADAM and AMSGrad
    t <- 0
    m <- rep(0, length(beta.init))
    v <- rep(0, length(beta.init))
    vHat <- rep(0, length(beta.init))
    
    if (bootstrap){ # if replacement use sample with repalcement
      sample.indices.all <- sample(1:num.rows.big, num.rows.big , replace = T)
    }else{ # if non-bootstrap, it samples without replacement
      sample.indices.all <- sample(1:num.rows.big, num.rows.big , replace = F)
    }
    
    # divides data indices to chunks of indices if data size is greater than chunk size
    num.rows.chunk <- strata.size*floor(num.rows.chunk/strata.size)
    num.round <- num.rows.big/num.rows.chunk
    num.round.vec <- cumsum(c(0, rep(num.rows.chunk, floor(num.round)),
                              if((num.round %% 1)>0){ceiling((num.round %% 1)*num.rows.chunk)}))
    # a loop over epochs
    for (n_e in 1:num.epoch){
      # Shuffles indices for new epoch
      sample.indices <- sample(sample.indices.all, num.rows.big, replace = F)
      
      # goes through all chunks of data for an epoch
      for (i.chunk in 1:(length(num.round.vec)-1)){
        # Indices for each chunk of data
        indices.chosen <- sample.indices[(num.round.vec[i.chunk]+1):(num.round.vec[i.chunk+1])]
        # Sub-data for a chunk of data
        sub.data <- as.matrix(big.data[indices.chosen, c(surv.indices, features.indices)])
        # Add a small jitter to time-to-event outcomes to deal with ties
        time.unique.sorted <- sort(unique(sub.data[,1]))
        sd.jitter <- 0.1*min(abs(time.unique.sorted[2:length(time.unique.sorted)]-
                                   time.unique.sorted[1:(length(time.unique.sorted)-1)]))
        sub.data[,1] <- sub.data[,1]+rnorm(NROW(sub.data), mean=0, sd=sd.jitter)
        
        
        # update the coefficients and parameters for one chunk of data
        oneChunkResult <- oneChunkC(sub.data, beta.init, beta.type,
                                    strata.size, batch.size,
                                    t, m, v, vHat, lr.const, lr.tau,
                                    opt.method, norm.method,
                                    b1, b2, eps,
                                    lambda, alpha,
                                    features.mean, features.sd)
        
        # update parameters after each chunk of data
        beta.init <- oneChunkResult$beta
        beta.ave <- oneChunkResult$betaAve
        t <- oneChunkResult$t
        tAve <- oneChunkResult$tAve
        m <- oneChunkResult$m
        v <- oneChunkResult$v
        vHat <- oneChunkResult$vHat
        
        if (beta.type == "averaged"){ # calculate average over iterates
          beta.norm <- ((t-tAve)*beta.norm+tAve*oneChunkResult$betaAve)/t
        }else{ # calculate single estimate
          beta.norm <- beta.init
        }
      } # for chunks
    } # for epoch
    return(beta.norm)
  }
  #######################################################################
  
  
  
  #######################################################################
  ## One iterate of plugin approach for a single subject
  do.plugin <- function(k, big.data=big.data, num.rows.chunk, num.rows.big,
                        norm.method, beta.hat,
                        strata.size, num.sample.strata,
                        surv.indices, features.indices, features.mean,
                        features.sd){
    
    # number of iteration of data needed for total sample strata
    num.round <- num.sample.strata/floor((num.rows.big-1)/(strata.size-1))
    num.round.vec <- c(rep(floor((num.rows.big-1)/(strata.size-1)),floor(num.round)),
                       if((num.round %% 1)>0){ceiling((num.round %% 1)*floor((num.rows.big-1)/(strata.size-1)))})
    
    # exclude observation k from whole cohort
    indices.Not.k <- (1:num.rows.big)[-k]
    indices.stratas <- NULL
    for (i in 1:length(num.round.vec)){
      indices.stratas <- c(indices.stratas, sample(indices.Not.k, num.round.vec[i]*(strata.size-1), replace = F))
    }
    
    r.cum <- 0
    h.cum <- 0
    
    # maps chunk size to a number that is divisible by starta.size-1
    num.rows.chunk <- (strata.size-1)*floor(num.rows.chunk/(strata.size-1))
    num.round <- length(indices.stratas)/num.rows.chunk
    num.round.vec <- cumsum(c(0, rep(num.rows.chunk, floor(num.round)),
                              if((num.round %% 1)>0){ceiling((num.round %% 1)*num.rows.chunk)}))
    # a loop over chunks of data
    for (i.chunk in 1:(length(num.round.vec)-1)){
      # check if we are at the last chunk and that we have enough data
      # Indices for a chunk of data
      indices.chosen <- indices.stratas[(num.round.vec[i.chunk]+1):(num.round.vec[i.chunk+1])]
      # Sub-data for a chunk of data
      sub.data <- as.matrix(big.data[c(k, indices.chosen), c(surv.indices, features.indices)])
      oneObsResults <- oneObsPlugingC(sub.data, beta.hat, strata.size, norm.method, features.mean, features.sd)
      r.cum <- r.cum + oneObsResults$Grad
      h.cum <- h.cum + oneObsResults$Hessian
    }
    r.cum <- r.cum/num.sample.strata
    h.cum <- h.cum/num.sample.strata
    r.cum <- matrix(r.cum, ncol = 1)%*%matrix(r.cum, nrow = 1)
    # return H and V matrices corresponding to observation k
    list(r.cum=r.cum, h.cum=h.cum)
  }
  #######################################################################
  
  
  #######################################################################
  # initializes beta as 0 if not specified in advance
  if (is.null(beta.init)){
    beta.init <- matrix(0,length(features.indices),1)
  }
  
  # If alpha specified but not lambda, it estimates maximum lambda and then makes a vector of lambdas
  if (!is.null(alpha) & is.null(lambda)){
    
    # Makes sure alpha is greater than or equal to 0
    if (alpha < 0){
      warning("alpha < 0; set to 0")
      alpha <- 0
    }
    # Makes sure alpha is less than or equal to 1
    if (alpha > 1){
      warning("alpha > 1; set to 1")
      alpha <- 1
    }
    
    # Estimates maximum lambda for which all coefiicients are 0 (it scales up with factor lambda.scale)
    lam.max <- lambda.max(big.data, strata.size, num.rows.big, norm.method, num.strata.lambda,
                          features.indices, surv.indices) * lambda.scale
    # If number of features is less than number of subjects, consider vector of smaller lambdas
    if (length(features.indices) < num.rows.big){
      lam.min <- 0.01*lam.max
    }else{
      lam.min <- 0.0001*lam.max
    }
    
    # Creates a vector of lambda with maximum element of lam.max
    lambdaAll <- round(lam.max * (lam.min/lam.max)^seq(0,1,l=nlambda), 10)
    # Initializes a matrix with each column is for each lambda
    beta.hat <- matrix(0, length(beta.init), length(lambdaAll))
    colnames(beta.hat) <- paste0("lambda=", lambdaAll)
    rownames(beta.hat) <- sub.col.names
    
    # Estimates coefficients for all specified lambdas given by lambdaAll
    for (l in 1:length(lambdaAll)){
      results.hat <- do.one(big.data=big.data, num.rows.chunk, num.rows.big, norm.method, opt.method,
                            beta.init, beta.type="single",
                            lr.const, lr.tau, strata.size, batch.size,
                            num.epoch, b1, b2, eps,
                            lambda=lambdaAll[l], alpha, bootstrap=F,
                            features.indices, surv.indices,
                            features.mean, features.sd)
      
      # Uses estimated coefficients from a bigger lambda for smaller lambdas
      beta.init <- results.hat
      beta.hat[,l] <- results.hat
    }
    
    # Scales back the coefficients is "scaling" or "normalization has been used"
    beta.hat[features.sd!=0,] <- beta.hat[features.sd!=0,]/matrix(rep(features.sd[features.sd!=0], length(lambdaAll)), nrow(beta.hat),
                                                                  length(lambdaAll), byrow = F)
    beta.hat.exp <- exp(beta.hat)
    lambda <- lambdaAll
  }else if (!is.null(alpha) & !is.null(lambda)){
    # Makes sure alpha is greater than or equal to 0
    if (alpha < 0){
      warning("alpha < 0; set to 0")
      alpha <- 0
    }
    # Makes sure alpha is less than or equal to 1
    if (alpha > 1){
      warning("alpha > 1; set to 1")
      alpha <- 1
    }
    # Makes sure lambda is greater than or equal to 0
    if (lambda < 0){
      warning("lambda < 0; set to 0")
      lambda <- 0
    }
    
    # Estimates coefficients for speficed pair of (alpha, lambda)
    beta.hat <- do.one(big.data=big.data, num.rows.chunk, num.rows.big, norm.method, opt.method,
                       beta.init, beta.type="single",
                       lr.const, lr.tau, strata.size, batch.size,
                       num.epoch, b1, b2, eps,
                       lambda, alpha, bootstrap=F,
                       features.indices, surv.indices,
                       features.mean, features.sd)
    
    # Scales back the coefficients is "scaling" or "normalization has been used"
    beta.hat[features.sd!=0] <- beta.hat[features.sd!=0]/features.sd[features.sd!=0]
    names(beta.hat) <- sub.col.names
    beta.hat.exp <- exp(beta.hat)
  }else{ # No regularization
    alpha <- 0
    lambda <- 0
    # Estimates coefficients without regularization
    beta.hat <- do.one(big.data=big.data, num.rows.chunk, num.rows.big, norm.method, opt.method,
                       beta.init, beta.type,
                       lr.const, lr.tau, strata.size, batch.size,
                       num.epoch, b1, b2, eps,
                       lambda=0, alpha=0, bootstrap=F,
                       features.indices, surv.indices,
                       features.mean, features.sd)
    
    if (inference.method == "bootstrap"){ # Uses bootstrap approach for constructing confidence interval
      # Initializes coefficients for non-bootstrap sample and bootstrap resamples
      if (!parallel.flag){ # If parallel computation has not been requested
        beta.boot <- matrix(0, length(beta.hat), num.boot)
        for (i in 1:num.boot){
          beta.boot[, i] <- do.one(big.data=big.data, num.rows.chunk, num.rows.big, norm.method, opt.method=boot.method,
                                   beta.hat, beta.type,
                                   lr.const=lr.const.boot, lr.tau=lr.tau.boot, strata.size, batch.size,
                                   num.epoch.boot, b1, b2, eps,
                                   lambda=0, alpha=0, bootstrap=T,
                                   features.indices, surv.indices,
                                   features.mean, features.sd)
        }
        
      }else{ # If parallel computation has been requested
        # Uses all cores of your system if it has not been specified
        if (is.null(num.cores)){
          num.cores <- detectCores()
        }
        # Registers requested number of cores
        # cl <- makeCluster(num.cores)
        registerDoParallel(cores = num.cores)
        beta.boot <- foreach(i=1:num.boot, .combine = cbind) %dopar% {
          do.one(big.data=big.data, num.rows.chunk, num.rows.big, norm.method, opt.method=boot.method,
                 beta.hat, beta.type,
                 lr.const=lr.const.boot, lr.tau=lr.tau.boot, strata.size, batch.size,
                 num.epoch.boot, b1, b2, eps,
                 lambda=0, alpha=0, bootstrap=T,
                 features.indices, surv.indices,
                 features.mean, features.sd)
        }
      }
      
      # Estimates quantiles and then CI
      quantiles.boot <- apply((beta.boot-matrix(rep(beta.hat, num.boot),
                                                length(beta.hat), num.boot, byrow = F)), 1,
                              function(x) quantile(x, probs = c((sig.level/2),
                                                                (1-sig.level/2)), na.rm = T))
      beta.hat <- cbind(beta.hat,
                        beta.hat-quantiles.boot[2,],
                        beta.hat-quantiles.boot[1,])
      rownames(beta.hat) <- sub.col.names
      colnames(beta.hat) <- c("estimate", paste0("lower ", (1-sig.level),"%CI"),  paste0("upper ", (1-sig.level),"%CI"))
      # Scales back the coefficients is "scaling" or "normalization has been used"
      beta.hat[features.sd!=0,1:3] <- beta.hat[features.sd!=0,1:3]/matrix(rep(features.sd[features.sd!=0], 3), nrow(beta.hat), 3, byrow = F)
      beta.hat.exp <- exp(beta.hat)
    } else if (inference.method == "plugin") { # Uses plugin approach for constructing confidence interval
      v.hat <- h.hat <- 0
      if (!parallel.flag){
        
        for (k in 1:num.rows.big){
          results.vh <- do.plugin(k, big.data=big.data, num.rows.chunk, num.rows.big,
                                  norm.method, beta.hat,
                                  strata.size, num.sample.strata,
                                  surv.indices, features.indices, features.mean,
                                  features.sd)
          
          v.hat <- v.hat+results.vh$r.cum
          h.hat <- h.hat+results.vh$h.cum
        }
        
      }else{
        # Uses all cores of your system if it has not been specified
        if (is.null(num.cores)){
          num.cores <- detectCores()
        }
        # Registers requested number of cores
        #cl <- makeCluster(num.cores)
        registerDoParallel(cores = num.cores)
        
        grad.hes <- foreach(k=1:num.rows.big) %dopar% {
          do.plugin(k, big.data=big.data, num.rows.chunk, num.rows.big,
                    norm.method, beta.hat,
                    strata.size, num.sample.strata,
                    surv.indices, features.indices, features.mean,
                    features.sd)
        }
        
        for (i in 1:length(grad.hes)){
          v.hat <- v.hat+grad.hes[[i]]$r.cum
          h.hat <- h.hat+grad.hes[[i]]$h.cum
        }
      }
      
      v.hat <- strata.size^2 * v.hat / num.rows.big
      h.hat <- h.hat / num.rows.big
      
      # Estimates covariance matrix and then CI
      Sigma.hat <- solve(h.hat) %*% v.hat %*% solve(h.hat)
      se.hat <- sqrt(diag(Sigma.hat)/num.rows.big)
      t.stat <- (beta.hat-beta0)/se.hat
      p.value <- 2*pt(-abs(t.stat),df=num.rows.big-1)
      beta.hat <- cbind(beta.hat, beta.hat - qt(1-(sig.level/2), df=num.rows.big) * se.hat,
                        beta.hat + qt(1-(sig.level/2), df=num.rows.big) * se.hat,
                        t.stat, p.value)
      colnames(beta.hat) <- c("estimate", paste0("lower ", (1-sig.level),"%CI"),  paste0("upper ", (1-sig.level),"%CI"), "z", "p-value")
      rownames(beta.hat) <- sub.col.names
      # Scales back the coefficients is "scaling" or "normalization has been used"
      beta.hat[features.sd!=0,1:3] <- beta.hat[features.sd!=0,1:3]/
        matrix(rep(features.sd[features.sd!=0], 3), nrow(beta.hat), 3, byrow = F)
      beta.hat.exp <- beta.hat
      beta.hat.exp[,c(1,2,3)] <- exp(beta.hat[,c(1,2,3)])
    }else{ # If constructing CI has not been asked
      names(beta.hat) <- sub.col.names
      beta.hat[features.sd!=0] <- beta.hat[features.sd!=0]/features.sd[features.sd!=0]
      beta.hat.exp <- exp(beta.hat)
    }
    #######################################################################
  }
  out <- NULL
  out$coef <- beta.hat
  out$coef.exp <- beta.hat.exp
  out$lambda <- lambda
  out$alpha <- alpha
  out$features.mean <- features.mean
  out$features.sd <- features.sd
  out$call <- match.call()
  class(out) <- "bigSurvSGD"
  out
}
#' A S3 function to print output
#' @rdname bigSurvSGD
#' @param x a 'bigSurvSGD' object
#' @exportMethod  
#' S3method(bigSurvSGD, "bigSurvSGD", met="print.bigSurvSGD")
print.bigSurvSGD <- function(x, ...){
  cat("Call:\n")
  print(x$call)
  cat("\nCoefficients (log hazards ratio)\n")
  if(ncol(x$coef)==5){
    X5 <- matrix(NA, nrow(x$coef),1)
    colnames(X5) < "p-value"
    for(i in 1:nrow(x$coef)){
      if(x$coef[i,5]<2e-16){
        X5[i] <- "<2e-16 ***"
      }else if(x$coef[i,5]<0.001){
        X5[i] <- "<0.001 ***"
      }else if(x$coef[i,5]<0.01){
        X5[i] <- "<0.01 **"
      }else if(x$coef[i,5]<0.05){
        X5[i] <- paste0(round(x$coef[i,5],2), " *")
      }else{
        X5[i] <- paste0(round(x$coef[i,5],2))
      }
    }
    X <- data.frame(cbind(round(x$coef[,1:4],4), X5))
    colnames(X)[5] <- "p-value'"
    print(X)
  }else{
    print(data.frame(round(x$coef,4)))
  }
  cat("\nCoefficients (hazards ratio)\n")
  if(ncol(x$coef.exp)==5){
    X <- data.frame(cbind(round(x$coef.exp[,1:4],4), X5))
    colnames(X)[5] <- "p-value"
    print(X)
  }else{
    print(data.frame(round(x$coef.exp,4)))
  }
}
#' A S3 function to plot coefficients against lambdas
#' @rdname bigSurvSGD
#' @param x a 'bigSurvSGD' object
#' @param ... additional argument used
#' @exportMethod 
#' S3method(bigSurvSGD, "bigSurvSGD", met="plot.bigSurvSGD") 
plot.bigSurvSGD <- function(x, ...){
  plot(x$lambda, x$coef[1,], xlab = "lambda", ylab = "coefficients", type = "l", col=1,
       ylim = c(min(x$coef), max(x$coef)))
  for(l in 2:nrow(x$coef)){
    lines(x$lambda, x$coef[l,], lty=1, col=l)
  }
}

