library(dHSIC)
library(mgcv)
library(glmnet)
library(InvariantCausalPrediction)
library(nonlinearICP)
library(gptk)
library(HCR)

#' check if integer(0)
#'
#' @param x A single value
#' @return Boolean
#'
#' @examples
#' nums <- 2
#' is_integer0(nums)
#'
#' @export
is_integer0 <- function(x) {
  is.integer(x) && length(x) == 0L
}

# run RESIT with Tübingen CEP
# Get the working directory
wd <- getwd()
source(file.path(wd, 'RESIT.R'))
source(file.path(wd, 'RESIT_fitting.R'))
source(file.path(wd, 'RESIT_indtests.R'))
source("codeANM/code/startups/startupLINGAM.R", chdir = TRUE)
source("codeANM/code/startups/startupICML.R", chdir = TRUE)
source("codeANM/code/startups/startupBF.R", chdir = TRUE)
source("codeANM/code/startups/startupGDS.R", chdir = TRUE)
#source("codeANM/code/startups/startupGES.R", chdir = TRUE)
source("codeANM/code/startups/startupPC.R", chdir = TRUE)
source("codeANM/code/startups/startupScoreSEMIND.R", chdir = TRUE)
pars <- list(regr.method = train_linear, regr.pars = list(), indtest.method = indtestHsic, indtest.pars = list())

ANM <- function(data, model) {
  result <- RESIT(data,model=model,force_answer = TRUE, output = FALSE)
  x_y = result[,1][2]
  y_x = result[,2][1]
  if (is.null(x_y)) return ("no_decision")
  if (x_y == 1 & y_x == 1) return("no_decision")
  if (x_y == 1) return("x->y")
  else if (y_x == 1) return("y->x")
  else return("no_decision")
}


in_str <- function(sub_str, str) {
  return(length(grep(sub_str, str, value=TRUE)) != 0)
}

check_result <- function(model_output, ground_truth) {
  if ((in_str("x -> y", ground_truth) | in_str("x->y", ground_truth) | in_str("x --> y", ground_truth) | in_str("x-->y", ground_truth) | in_str("x - - > y", ground_truth)) & (model_output == "x->y")) {
    return(TRUE)
  }
  if ((in_str("y -> x", ground_truth) | in_str("y->x", ground_truth) | in_str("y --> x", ground_truth) | in_str("y-->x", ground_truth) | in_str("y - - > x", ground_truth) | in_str("x <- y", ground_truth)) & (model_output == "y->x")) {
    return(TRUE)
  }
  return(FALSE)
}


correct = 0
not_correct = 0
no_decision = 0
simulation = FALSE
dataset = "Abalone"
method = 'HCR'
if (simulation == TRUE) {
  directory = "../simulations/add_mult/linear_continuous/2_2/"
}
if (dataset == 'Food') {
  directory = "../real_world_data/food_intolerances/"
} else if (dataset == 'Abalone') {
  directory = "../real_world_data/abalone/"
} else if (dataset == 'Bridge') {
  directory = "../real_world_data/bridge_data/"
} else if (dataset == 'Extern') {
  directory = "../simulations/extern/"
} else if (dataset == 'CEP') {
  directory = "../pairs/"
}
for (file in list.files(file.path(wd, directory))) {
  if ((simulation == FALSE & (in_str("pair0", file) | in_str(".csv", file)) & in_str("_des", file) == FALSE) | (simulation == TRUE)) {
    result = tryCatch({
      print(file)
      #file="purpose_type.csv"
      #file="pair93.csv"
      
      if (dataset == 'CEP' | dataset == 'own') {
        ce_pair_data <- read.csv(file.path(wd, paste(directory,file, sep="")), sep=" ", header=FALSE)
      } else {
        ce_pair_data <- read.csv(file.path(wd, paste(directory,file, sep="")), sep=";", header=FALSE)
      }
      names(ce_pair_data) = c('X', 'Y')
      ce_pair_data = ce_pair_data[order(ce_pair_data$X),]
      ce_pair_data = as.data.frame(apply(ce_pair_data[, c('X', 'Y')], 2, function(x) (x - mean(x))/(sd(x))))
      ce_pair_data[is.na(ce_pair_data)] = 0
      if (simulation == TRUE | dataset == "Bridge" | dataset == "Food" | dataset=="Abalone" | dataset=="Extern") {
        ce_pair_desc = "x->y"
      } else {
        ce_pair_desc <- tolower(readLines(file.path(wd, gsub(".txt", "_des.txt", paste("../pairs/",file, sep="")))))
      }
      if (method == 'HCR') {
        r1 <- HCR(ce_pair_data[, 'X'], ce_pair_data[, 'Y'], is_anm=TRUE)
        r2 <- HCR(ce_pair_data[, 'Y'], ce_pair_data[, 'X'], is_anm=TRUE)
        res_ = "no_decision"
        if (r1$score > r2$score) {
          res_ = "x->y"
        } else if (r1$score < r2$score) {
          res_ = "y->x"
        }
      }
      #res <- ICML(cbind(ce_pair_data[, 'X'], ce_pair_data[, 'Y']), model = train_gp, indtest = indtestHsic, output = FALSE)
      #res <- GDS(cbind(ce_pair_data[, 'X'], ce_pair_data[, 'Y']), "SEMIND", pars, check = "checkUntilFirst", output = FALSE, kvec = c(10000), startAt = "emptyGraph")$Adj
      #res <- lingamWrap(cbind(ce_pair_data[, 'X'], ce_pair_data[, 'Y']))$Adj
      #res <- BruteForce(cbind(ce_pair_data[, 'X'], ce_pair_data[, 'Y']), "SEMIND", pars, output = FALSE)$Adj
      else if (method == 'ICP') {
        n=length(ce_pair_data[, 'X'])
        n_x = floor(n/2)
        ExpInd <- as.factor(c(rep(1,n_x),rep(2,n-n_x)))
        res = list()
        resy = list()
        tryCatch({
        res = ICP(ce_pair_data[, 'X'], ce_pair_data[, 'Y'], ExpInd)
        }, error = function(e){
          # do nothing
        })
        tryCatch({
        resy = ICP(ce_pair_data[, 'Y'], ce_pair_data[, 'X'], ExpInd)
        }, error = function(e){
          # do nothing
        })
        res_ = "no_decision"
        if (length(res) != 0 &&
           (length(res$acceptedSets) == 2 && res$acceptedSets[[2]] == 1 && length(resy$acceptedSets) == 0) || 
           (length(res$acceptedSets) == 1 && is_integer0(res$acceptedSets[[1]]) == FALSE && res$acceptedSets[[1]] == 1 && length(resy$acceptedSets) == 0)) {
          res_ = "x->y"
        } else if (length(resy) != 0 && (length(resy$acceptedSets) == 2 && resy$acceptedSets[[2]] == 1 && length(res$acceptedSets) == 0) ||
                   (length(resy$acceptedSets) == 1 && is_integer0(resy$acceptedSets[[1]]) == FALSE && resy$acceptedSets[[1]] == 1 && length(res$acceptedSets) == 0)) {
          res_ = "y->x"
        }
      }
      else if (method == 'nonICP') {
        n=length(ce_pair_data[, 'X'])
        n_x = floor(n/2)
        ExpInd <- as.factor(c(rep(1,n_x),rep(2,n-n_x)))
        res = 0
        res = nonlinearICP(cbind(ce_pair_data[, 'X']), ce_pair_data[, 'Y'], ExpInd)
        resy = nonlinearICP(cbind(ce_pair_data[, 'Y']), ce_pair_data[, 'X'], ExpInd)
        #res = ICP(ce_pair_data[, 'X'], ce_pair_data[, 'Y'], ExpInd)
        res_ = "no_decision"
        if (length(res) != 0 &&
            (length(res$acceptedSets) == 2 && res$acceptedSets[[2]] == 1 && length(resy$acceptedSets) == 0) || 
            (length(res$acceptedSets) == 1 && is_integer0(res$acceptedSets[[1]]) == FALSE && res$acceptedSets[[1]] == 1 && length(resy$acceptedSets) == 0)) {
          res_ = "x->y"
        } else if (length(resy) != 0 && (length(resy$acceptedSets) == 2 && resy$acceptedSets[[2]] == 1 && length(res$acceptedSets) == 0) ||
                   (length(resy$acceptedSets) == 1 && is_integer0(resy$acceptedSets[[1]]) == FALSE && resy$acceptedSets[[1]] == 1 && length(res$acceptedSets) == 0)) {
          res_ = "y->x"
        }
      }
      
#      if (res[1,2] == 1) {
#       res_ = "x->y"
#      }
#      if (res[2,1] == 1) {
#       res_ = "y->x"
#      }
      #Timino_res = timino_pairwise(ce_pair_data[, 'X'], ce_pair_data[, 'Y'], alpha = 0.05, max_lag = 2, instant=1, model=traints_gam, indtest = indtestts_crosscov)
      #anm_res = ANM(ce_pair_data, train_GAMboost)
      if (res_ == "no_decision") {
        no_decision = no_decision + 1
        print("no decision")
      } else {
        if (check_result(res_, ce_pair_desc) == TRUE) {
          correct = correct + 1
          print("correct")
        } else {
          not_correct = not_correct + 1
          print("not correct")
        }
      }
    }, error = function(e) {
      print("failed for ")
      print(e)
      print(file)
    })
  }
}

total_number = correct + not_correct + no_decision
print(paste((correct / total_number*100.0)," (",correct,",",not_correct,",",no_decision,"/",total_number,")",sep=""))
print(paste("correct:",correct, "(",(correct / total_number*100.0), "%)" ))
print(paste("not correct:",not_correct, "(",(not_correct / total_number*100.0), "%)" ))
print(paste("not decision:",no_decision, "(",(no_decision / total_number*100.0), "%)" ))
total_number

file="nonlinear_discrete_extern_1.csv"
ce_pair_data <- read.csv(file.path(wd, paste(directory,file, sep="")), sep=";", header=FALSE)
names(ce_pair_data) = c('X', 'Y')
scatter.smooth(ce_pair_data[,'X'], ce_pair_data[,'Y'])


data=simuXY(sample_size=2000)
r1=HCR(data$X,data$Y)
r2=HCR(data$Y,data$X)
r1$score < r2$score
r1$score
r2$score
# The canonical hidden representation
unique(r1$data[,c("X","Yp")])
# The recovery of hidden representation
data
unique(data.frame(data$X,data$Yp))


X <- rnorm(100)
Y <- rnorm(50)
Y0 <- rnorm(25)
Y1 <- rnorm(25)
cor(X,c(Y,Y0,Y1))
cor(X[76:100],Y0)
