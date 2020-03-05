library(dHSIC)
library(mgcv)
library(glmnet)

# run RESIT with Tübingen CEP
# Get the working directory
wd <- getwd()
source(file.path(wd, 'RESIT.R'))
source(file.path(wd, 'RESIT_fitting.R'))
source(file.path(wd, 'RESIT_indtests.R'))

ANM <- function(data, model) {
  result <- RESIT(data,model=model,force_answer = FALSE, output = FALSE)
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
for (file in list.files(file.path(wd, "../pairs/"))) {
  if (in_str("pair0", file) & in_str("_des", file) == FALSE) {
    result = tryCatch({
      print(file)
      
      ce_pair_data <- read.csv(file.path(wd, paste("../pairs/",file, sep="")), sep=" ", header=FALSE)
      names(ce_pair_data) = c('X', 'Y')
      ce_pair_data = as.data.frame(apply(ce_pair_data[, c('X', 'Y')], 2, function(x) (x - mean(x))/(sd(x))))
      ce_pair_desc <- tolower(readLines(file.path(wd, gsub(".txt", "_des.txt", paste("../pairs/",file, sep="")))))
      
      anm_res = ANM(ce_pair_data, train_linear)
      if (anm_res == "no_decision") {
        no_decision = no_decision + 1
        print("no decision")
      } else {
        if (check_result(anm_res, ce_pair_desc) == TRUE) {
          correct = correct + 1
          print("correct")
        } else {
          not_correct = not_correct + 1
          print("not correct")
        }
      }
    }, error = function(e) {
      print("failed for ")
      print(file)
    })
  }
}

total_number = correct + not_correct + no_decision
print(paste("correct:",correct, "(",(correct / total_number*100.0), "%)" ))
print(paste("not correct:",not_correct, "(",(not_correct / total_number*100.0), "%)" ))
print(paste("not decision:",no_decision, "(",(no_decision / total_number*100.0), "%)" ))

