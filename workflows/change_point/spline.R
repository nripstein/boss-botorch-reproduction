
args <- commandArgs(TRUE)
d <- read.csv(args[1])
for (b in c(20,50)) {
  x <- (d$alpha_1[1:b]-.1)/9.8
  y <- d$objective[1:b]; y <- y-mean(y)
  if(anyDuplicated(x) || any(!is.finite(y))) stop('Invalid spline observations')
  fit <- npreg::ss(x=x,y=y,df=length(unique(x)),m=2,all.knots=TRUE)
  u <- seq(0,1,length.out=1000)
  z <- as.numeric(predict(fit,x=u)$y)
  p <- exp(z-max(z))
  normalizer <- sum((p[-1]+p[-length(p)])*diff(u)/2)
  write.csv(data.frame(x=.1+9.8*u,pos=p/normalizer/9.8),
            file.path(args[2],paste0('reproduced_b',b,'.csv')),row.names=FALSE)
}
