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

//' @title Calculates the gradient and Hessian corresponding to one individual
//' @name oneObsPlugingC

// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
List oneObsPlugingC(NumericMatrix subDataa, NumericVector Beta,
int strata_size, std::string norm_method, NumericVector features_mean, 
NumericVector features_sd) {

	int p = Beta.size();
	NumericVector time2event(strata_size);
	NumericVector statusdeath(strata_size);
	NumericMatrix Cov(strata_size, p);
	NumericMatrix subData = clone(subDataa);
	int start_index;
	

	NumericVector Grad(p);
	NumericMatrix Hessian(p);

	
	NumericVector xb;
	NumericVector expxb;
	
	// pre-processes data
	if (norm_method!="none"){
		for (int i=2; i<subData.ncol(); ++i){
			// centers features
			if (norm_method=="center" || norm_method=="standardize"){
				for (int k=0; k<subData.nrow(); ++k){
					subData(k, i) = subData(k, i) - features_mean[(i-2)];
				}
			}
			if (norm_method=="scale" || norm_method=="standardize"){
				if (features_sd[(i-2)]!=0){
					for (int k=0; k<subData.nrow(); ++k){
						subData(k,i) = subData(k,i)/features_sd[(i-2)];
					}
				}
			}
		}
	}

	int num_strata = (int) ((subData.nrow()-1)/(strata_size-1));
	for (int i_strata=0; i_strata<num_strata; i_strata++){
		start_index = i_strata*(strata_size-1)+1;
		for (int j=0; j<(strata_size-1); ++j){
			time2event[(j+1)] = subData((start_index+j), 0);
			statusdeath[(j+1)] = subData((start_index+j), 1);
			for (int k=2; k<subData.ncol(); ++k){
				Cov((j+1), k-2) = subData(start_index+j, k);
			}
		}

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

		NumericVector Num(p);
		NumericVector Den(1);
		NumericMatrix NumH(p);
		
	

		for (int i=0; i<strata_size; ++i) {
			xb = 0;
			expxb =0;
			for (int j=0; j<p; ++j){
				xb = xb + Cov(c[i],j)*Beta[j];
			} 
			expxb = exp(xb);
			Den[0] = Den[0] + expxb[0];
			Num = Num + expxb[0]*Cov( c[i], _);
			Grad = Grad + (Cov(c[i], _)-Num/Den[0])*statusdeath[c[i]];
			for (int k=0; k<p; ++k){
				for (int l=0; l<=k; ++l){
					NumH(k, l) = NumH(k, l) + expxb[0]*Cov(c[i], k)*Cov(c[i], l);
					Hessian(k, l) = Hessian(k, l) + (Num[k]*Num[l]/Den[0]/Den[0]-NumH(k, l)/Den[0])*statusdeath[c[i]];
					Hessian(l, k) = Hessian(k, l);	
				}
			}
		}
	}

	List out;
	out["Grad"] = Grad;
	out["Hessian"] = Hessian;
	out["numS"] = num_strata;
	return out;
}
