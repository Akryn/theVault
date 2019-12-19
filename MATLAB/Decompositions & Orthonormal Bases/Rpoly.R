# NOTE: To obtain matrices Q and R from the output of R's qr() function (say, called QR), we must use qr.Q(QR) and qr.R(QR) respectively.

A = 0:20
outputA = poly(A, 3)

Astar = seq(0, 20, by = 0.5)
outputAstar = stats:::predict.poly(outputA, Astar)
# Correlations appear between the linear and cubic terms in outputAstar that didn't exist in outputA.

rVec = 0:20
cVec = 10:40
R = matrix(rep(rVec, length(cVec)), length(rVec)) # Changes over rows
C = matrix(rep(cVec, each = length(rVec)), length(rVec)) # Changes over columns
RC = cbind(c(R), c(C))
outputB = poly(RC, 2)

rVecstar = seq(0, 20, by = 0.5)
cVecstar = seq(10, 40, by = 0.5)
Rstar = matrix(rep(rVecstar, length(cVecstar)), length(rVecstar)) # Changes over rows
Cstar = matrix(rep(cVecstar, each = length(rVecstar)), length(rVecstar)) # Changes over columns
RCstar = cbind(c(Rstar), c(Cstar))
outputBstar = stats:::predict.poly(outputB, RCstar)
# Correlations appear between the linear and cubic terms in outputAstar that didn't exist in outputA.