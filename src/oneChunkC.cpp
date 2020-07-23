#include <Rcpp.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <stdio.h>
#include <math.h>
#include <numeric>
#include <algorithm>

using namespace Rcpp;
using namespace std;

//' @title Updates the coefficients based on one pass of data
//' @name oneChunkC

// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
List oneChunkC(NumericMatrix subData, NumericVector Beta, std::string beta_type,
int strata_size, int batch_size,
int t, NumericVector m, NumericVector v, NumericVector vHat, double lr_const, double lr_tau, 
std::string opt_method, std::string norm_method, 
double b1, double b2, double eps, 
double lambda, double alpha,
NumericVector features_mean, NumericVector features_sd) {


	int p = Beta.size();
	NumericVector betaAve(p);
	NumericVector beta = clone(Beta);
	int tAve = 0;

	// pre-processes data
	if (norm_method!="none"){
		for (int i=2; i<subData.ncol(); ++i){
			// centers features
			if (norm_method=="center" || norm_method=="standardize"){
				subData(_, i) = subData(_, i) - features_mean[(i-2)];
			}
			if (norm_method=="scale" || norm_method=="standardize"){
				if (features_sd[(i-2)]!=0){
					subData(_,i) = subData(_,i)/features_sd[(i-2)];
				}
			}
		}
	}

	int num_mini = (int) (subData.nrow()/strata_size/batch_size);

	for (int i_mini=0; i_mini<num_mini; ++i_mini){
		t++;
		tAve++;
		double LR = lr_const/pow(t,lr_tau);
		NumericVector gt(p);
	
		for (int i_strata=0; i_strata<batch_size; i_strata++){
			int start_index = i_mini*strata_size*batch_size+i_strata*strata_size;
			NumericVector time2event(strata_size);
			NumericVector statusdeath(strata_size);
			NumericMatrix Cov(strata_size, p);
			for (int j=0; j<strata_size; ++j){
				time2event[j] = subData((start_index+j), 0);
				statusdeath[j] = subData((start_index+j), 1);

				for (int k=2; k<subData.ncol(); ++k){
					Cov(j, k-2) = subData(start_index+j, k);
				}

			}

			// calculates the gradient
			std::multimap<double, std::size_t> mm;
			for (std::size_t i = strata_size-1; (int)i !=-1 ; --i){
    				mm.insert({time2event[i], i});
			}

			std::vector<std::size_t> c;
			std::vector<int> d;

			for (const auto & kv : mm) {
    				c.push_back(kv.second);
			}
			std::reverse(c.begin(), c.end());	

			NumericVector Den(1);
			NumericVector Num(p);
			NumericVector xb;
			NumericVector expxb;
			
			if (std::accumulate(statusdeath.begin(),statusdeath.end(), 0.0)>=1){

				for (int i=0; i<strata_size; i++) {
					xb = 0;
					expxb =0;
					for (int j=0; j<p; j++){
						xb = xb + Cov(c[i],j)*beta[j];
					} 

					expxb = exp(xb);
					Den[0] = Den[0] + expxb[0];
				
					Num = Num + expxb[0]*Cov(c[i], _);
					gt = gt + (Cov(c[i], _)-Num/Den[0])*statusdeath[c[i]];	
				}
			}
		}

		// update based on different algorithms
		if (opt_method == "SGD"){
			beta = beta + LR*gt/batch_size;
		} else if (opt_method == "ADAM"){
			gt = gt/batch_size;
			m = b1*m + (1-b1)*gt;
			v = b2*v + (1-b2)*pow(gt,2);
			NumericVector mHat = m/(1-pow(b1,t));
              		vHat = v/(1-pow(b2,t));
              		beta = beta + LR*mHat/(pow(vHat,0.5)+eps);
		}else {
		  	gt = gt/batch_size;
              		m = b1*m + (1-b1)*gt;
			v = b2*v + (1-b2)*pow(gt,2);

			for (int i=0; i<p; ++i){
				vHat[i] = max(v[i], vHat[i]);
			}
              		beta = beta + LR*m/(pow(vHat,0.5)+eps);
		}
		// soft-threshold
		if (lambda != 0){
			for (int i=0; i<beta.size(); ++i){
				if(beta[i]>(LR*alpha*lambda)){
					beta[i] = (beta[i]-(LR*alpha*lambda))/(1+LR*(1-alpha)*lambda);
				} else if (beta[i]<(-LR*alpha*lambda)){
					beta[i] = (beta[i]+(LR*alpha*lambda))/(1+LR*(1-alpha)*lambda);
				} else {
					beta[i] = 0;
				}
			}
		}
		if (beta_type == "averaged"){
              		betaAve = ((tAve-1)*betaAve+beta)/tAve;
        	}

	}

	List out;
	out["beta"] = beta;
	out["betaAve"] = betaAve;
	out["t"] = t;
	out["tAve"] = tAve;
	out["m"] = m;
	out["v"] = v;
	out["vHat"] = vHat;
	return out;
}
