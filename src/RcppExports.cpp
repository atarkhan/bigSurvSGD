// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// lambdaMaxC
NumericVector lambdaMaxC(NumericMatrix subDataa, int strata_size, std::string norm_method, NumericVector features_mean, NumericVector features_sd);
RcppExport SEXP _bigSurvSGD_lambdaMaxC(SEXP subDataaSEXP, SEXP strata_sizeSEXP, SEXP norm_methodSEXP, SEXP features_meanSEXP, SEXP features_sdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type subDataa(subDataaSEXP);
    Rcpp::traits::input_parameter< int >::type strata_size(strata_sizeSEXP);
    Rcpp::traits::input_parameter< std::string >::type norm_method(norm_methodSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type features_mean(features_meanSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type features_sd(features_sdSEXP);
    rcpp_result_gen = Rcpp::wrap(lambdaMaxC(subDataa, strata_size, norm_method, features_mean, features_sd));
    return rcpp_result_gen;
END_RCPP
}
// oneChunkC
List oneChunkC(NumericMatrix subData, NumericVector Beta, std::string beta_type, int strata_size, int batch_size, int t, NumericVector m, NumericVector v, NumericVector vHat, double lr_const, double lr_tau, std::string opt_method, std::string norm_method, double b1, double b2, double eps, double lambda, double alpha, NumericVector features_mean, NumericVector features_sd);
RcppExport SEXP _bigSurvSGD_oneChunkC(SEXP subDataSEXP, SEXP BetaSEXP, SEXP beta_typeSEXP, SEXP strata_sizeSEXP, SEXP batch_sizeSEXP, SEXP tSEXP, SEXP mSEXP, SEXP vSEXP, SEXP vHatSEXP, SEXP lr_constSEXP, SEXP lr_tauSEXP, SEXP opt_methodSEXP, SEXP norm_methodSEXP, SEXP b1SEXP, SEXP b2SEXP, SEXP epsSEXP, SEXP lambdaSEXP, SEXP alphaSEXP, SEXP features_meanSEXP, SEXP features_sdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type subData(subDataSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Beta(BetaSEXP);
    Rcpp::traits::input_parameter< std::string >::type beta_type(beta_typeSEXP);
    Rcpp::traits::input_parameter< int >::type strata_size(strata_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type batch_size(batch_sizeSEXP);
    Rcpp::traits::input_parameter< int >::type t(tSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type m(mSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type v(vSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type vHat(vHatSEXP);
    Rcpp::traits::input_parameter< double >::type lr_const(lr_constSEXP);
    Rcpp::traits::input_parameter< double >::type lr_tau(lr_tauSEXP);
    Rcpp::traits::input_parameter< std::string >::type opt_method(opt_methodSEXP);
    Rcpp::traits::input_parameter< std::string >::type norm_method(norm_methodSEXP);
    Rcpp::traits::input_parameter< double >::type b1(b1SEXP);
    Rcpp::traits::input_parameter< double >::type b2(b2SEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type features_mean(features_meanSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type features_sd(features_sdSEXP);
    rcpp_result_gen = Rcpp::wrap(oneChunkC(subData, Beta, beta_type, strata_size, batch_size, t, m, v, vHat, lr_const, lr_tau, opt_method, norm_method, b1, b2, eps, lambda, alpha, features_mean, features_sd));
    return rcpp_result_gen;
END_RCPP
}
// oneObsPlugingC
List oneObsPlugingC(NumericMatrix subDataa, NumericVector Beta, int strata_size, std::string norm_method, NumericVector features_mean, NumericVector features_sd);
RcppExport SEXP _bigSurvSGD_oneObsPlugingC(SEXP subDataaSEXP, SEXP BetaSEXP, SEXP strata_sizeSEXP, SEXP norm_methodSEXP, SEXP features_meanSEXP, SEXP features_sdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type subDataa(subDataaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Beta(BetaSEXP);
    Rcpp::traits::input_parameter< int >::type strata_size(strata_sizeSEXP);
    Rcpp::traits::input_parameter< std::string >::type norm_method(norm_methodSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type features_mean(features_meanSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type features_sd(features_sdSEXP);
    rcpp_result_gen = Rcpp::wrap(oneObsPlugingC(subDataa, Beta, strata_size, norm_method, features_mean, features_sd));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bigSurvSGD_lambdaMaxC", (DL_FUNC) &_bigSurvSGD_lambdaMaxC, 5},
    {"_bigSurvSGD_oneChunkC", (DL_FUNC) &_bigSurvSGD_oneChunkC, 20},
    {"_bigSurvSGD_oneObsPlugingC", (DL_FUNC) &_bigSurvSGD_oneObsPlugingC, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_bigSurvSGD(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}