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

//' @title Calculates the maximum penalty coefficient lambda for which all 
//' coefficients become zero
//' @name lambdaMaxC

// [[Rcpp::plugins("cpp11")]]

// [[Rcpp::export]]
NumericVector lambdaMaxC(NumericMatrix subDataa, int strata_size, std::string norm_method,
NumericVector features_mean, NumericVector features_sd) {

	int p = subDataa.ncol()-2;
	NumericVector G(p);
	NumericMatrix subData = clone(subDataa);
	
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
	

	int num_strata = (int) (subData.nrow()/strata_size);
	for (int i_strata=0; i_strata<num_strata; i_strata++){
		int start_index = i_strata*strata_size;
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

		std::multimap<double, std::size_t> mm;
		for (std::size_t i = strata_size-1; (int)i !=-1 ; --i){
    			mm.insert({time2event[i],i});
		}
		std::vector<std::size_t> c;
		std::vector<int> d;
		for (const auto & kv : mm) {
    			c.push_back(kv.second);            // c: 2 1 4 3 0
		}
		std::reverse(c.begin(), c.end());	

		NumericVector Num(p);
		for (int i=0; i<strata_size; ++i) {
			for (int k=0; k<p; k++){
				Num[k] = Num[k] + Cov(c[i], k);
				G[k] = G[k] + (Cov(c[i], k)-Num[k]/(i+1))*statusdeath[c[i]];
			}
		}
	}
	return G;
}

