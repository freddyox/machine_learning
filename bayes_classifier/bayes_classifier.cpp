#include "TGraph.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TMath.h"
#include "TAxis.h"
#include "TH1D.h"
#include "TStyle.h"
#include "TH2D.h"
#include "TLegend.h"
#include "TVectorD.h"
#include "TMatrix.h"
#include "TLatex.h"
#include "TArrow.h"

#include <map>
#include <iostream>
#include <vector>
#include <string>

#define ADD_LABELS 1

////////////////////////////////////////////////////////////////////////////////
//          Bayesian classification algorithms
//                Obrecht - 10/10/2018
//
//  Note: I use TVectorD as a pseudo-structure for an (x1,x2) point, i.e. a
//  vector<TVectorD> is a vector of data points where each element has an
//  x1,x2 coordinate. TMatrixD holds the covariance matrix. We will generate
//  pseudo-training data in an x1-x2 space, which are the features in this case,
//  in order to learn about each of the distributions (extract mean/covariance).
//  Then we can calculate the probability that a new point belongs to a class.

////////////////////////////////////////////////////////////////////////////////
// Simple method to make TGraphs - returns pointer with all properties
//
TGraph* make_plot(std::vector<TVectorD> &data,  const char* name,
		  const char* xaxis, const char* yaxis,
		  int marker, int color, float markersize){
  int N = data.size();
  double x[N],y[N];
  for(int i=0; i<N; i++){
    x[i] = data[i](0);
    y[i] = data[i](1);
  }
  
  TGraph *gtemp = new TGraph(N,x,y);
  gtemp->SetTitle("");
  gtemp->SetName(name);
  gtemp->SetMarkerStyle(marker);
  gtemp->SetMarkerColor(color);
  gtemp->SetMarkerSize(markersize);
  gtemp->SetLineColor(color);
  gtemp->SetLineWidth(2);
  gtemp->GetXaxis()->SetTitle(xaxis);
  gtemp->GetYaxis()->SetTitle(yaxis);
  gtemp->GetXaxis()->CenterTitle();
  gtemp->GetYaxis()->CenterTitle();
  return gtemp;
}

////////////////////////////////////////////////////////////////////////////////
// Generate normally distributed data in two dimensions with correlation
//  denoted by rho. Set rho=0.0 if the distributions are to be independent,
//  i.e. normally distributed in x1 and x2 with no correlations.
//  The routine fills x1 and x2 arrays
//
void random_data(int N, int nvars, double *means, double *sig, double rho,
		 std::vector<TVectorD>& data){

  TRandom fRand(0);
  
  // Local variables for human-readability
  double mu1 = means[0];
  double mu2 = means[1];
  double sig1 = sig[0];
  double sig2 = sig[1];
  for(int i=0; i<N; i++){
    double z1 = fRand.Gaus(mu1, sig1); // normally independent in x1 direction
    double z2 = fRand.Gaus(mu2, sig2); // normally independent in x2 direction
    double x1_normal = z1; 
    double x2_normal = rho*z1 + sqrt(1.0 - rho*rho) * z2; // add x1/x2 correlation
    TVectorD pt(2);
    pt(0) = x1_normal;
    pt(1) = x2_normal;
    data.push_back(pt);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Utility function for printing vector<TVectorD>
//
void print(std::vector<TVectorD>& data){
  for(unsigned int i=0; i<data.size(); i++){
    printf("(%1.3f, %1.3f), ", data[i](0), data[i](1));
  }
  std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Get the mean vector, i.e. the mean in x1 and x2 directions
//
TVectorD get_means_from_data(std::vector<TVectorD>& data){
  TVectorD means(2);
  if( data.empty() ) {
    std::cout << "The data vector is empty, and you will have problems..." << std::endl;
    means(0) = 0.0;
    means(1) = 0.0;
    return means;
  }
  double sum_x1=0.0, sum_x2=0.0;
  for(unsigned int i=0; i<data.size(); i++){
    sum_x1 += data[i](0);
    sum_x2 += data[i](1);
  }
  means(0) = sum_x1 / data.size();
  means(1) = sum_x2 / data.size();
  return means;
}

////////////////////////////////////////////////////////////////////////////////
// Divide all components of matrix mat by norm
//
void norm_matrix(TMatrixD& mat, double norm){
  if( norm < 1.e-6 ) return;
  int nrow = mat.GetNrows();
  int ncol = mat.GetNcols();
  for(int row=0; row<nrow; row++){
    for(int col=0; col<ncol; col++){
      mat(row,col) /= norm;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Extract the covariance matrix from the data
//
TMatrixD get_sigmas_from_data(std::vector<TVectorD>& data, TVectorD& means){
  TMatrixD sigmas(2,2); // rows by cols
  sigmas.Zero();
  for(unsigned int i=0; i<data.size(); i++){
    sigmas(0,0) += pow(data[i](0) - means(0), 2.0);
    sigmas(0,1) += ( data[i](0) - means(0) )*( data[i](1) - means(1) );
    sigmas(1,0) += ( data[i](0) - means(0) )*( data[i](1) - means(1) );
    sigmas(1,1) += pow(data[i](1) - means(1), 2.0);
  }
  norm_matrix(sigmas, data.size());
  return sigmas;
}

////////////////////////////////////////////////////////////////////////////////
// Using the mean/cov from the data, calculate the probability to be at pt
//
double probability(TVectorD& mean, TMatrixD& cov, TVectorD& pt){
  TMatrixD sig_invert(2,2);
  double det = cov(0,0)*cov(1,1) - cov(0,1)*cov(1,0);
  sig_invert(0,0) =  cov(1,1)/det;
  sig_invert(0,1) = -cov(0,1)/det;
  sig_invert(1,0) = -cov(1,0)/det;
  sig_invert(1,1) =  cov(0,0)/det;
  
  double dx1 = pt(0) - mean(0);
  double dx2 = pt(1) - mean(1);
  double arg = dx1*(dx1*sig_invert(0,0) + dx2*sig_invert(1,0))
    + dx2*(dx1*sig_invert(0,1) + dx2*sig_invert(1,1) ); 
  double prob = 1.0 / (2.0*TMath::Pi() * sqrt(det) ) * TMath::Exp(-0.5*arg);
  return prob;
}

////////////////////////////////////////////////////////////////////////////////
// Spit out useful information about the Class
//
void class_properties(const char* classname, std::vector<TVectorD>& data){
  TVectorD means = get_means_from_data(data);
  TMatrixD sigmas = get_sigmas_from_data(data, means);
  printf("Class %s:\n", classname);
  printf("N = %lu\n", data.size());
  printf("mu   = (%1.5f, %1.5f) \n", means(0), means(1));
  printf("var  = (%1.5f, %1.5f) \n", sigmas(0,0), sigmas(1,1) );
  printf("sig  = (%1.5f, %1.5f) \n", sqrt(sigmas(0,0)), sqrt(sigmas(1,1)) );
  printf("corr = (%1.5f, %1.5f) \n", sigmas(0,1), sigmas(1,0) );
  return;
}

////////////////////////////////////////////////////////////////////////////////
// Build the contours based off the mean/covariance from the data
//
TH2D* build_gauss(const char* classname, std::vector<TVectorD> &data,
		  int nbins, double xlo, double xhi){
  if( data.empty() ) return NULL;
  
  TVectorD means = get_means_from_data(data);
  TMatrixD sigmas = get_sigmas_from_data(data, means);
  class_properties(classname, data);

  TH2D* htemp = new TH2D(Form("class_%s_2d",classname),"",
			 nbins, xlo, xhi, nbins, xlo, xhi);

  double dx = (xhi-xlo) / nbins; // incremental step in x
  double dy = dx;                // incremental step in y (same by default)
  xlo += 0.5*dx;                 // evaluate at center of bin, important

  double integral = 0.0;         // sum the probability distribution

  // And now let's scan through [0.0, 1.0] in x1 and x2 to get the function
   for(int xi=0; xi<nbins; xi++){
    for(int yi=0; yi<nbins; yi++){
      double xtemp = xlo + xi*dx;
      double ytemp = xlo + yi*dy;
      double pt[2] = {xtemp,ytemp};
      TVectorD point; point.Use(2,pt);
      double P = probability(means,sigmas,point);
      htemp->Fill(xtemp, ytemp, P/data.size());
      integral += P/data.size()/data.size();
    }
  }
   std::cout << "integral = " << integral << std::endl;
   return htemp;
}

////////////////////////////////////////////////////////////////////////////////
// Main driver script-code
//
void bayes_classifier(){
  gStyle->SetPalette(55);
  gStyle->SetOptStat(0);

  // Let's have three classes:
  std::vector<TVectorD> A,B,C;
  
  // Class A:
  int NClassA = 70;
  double rho = 0.25;
  double mu[]  = {0.3, 0.475};
  double sig[] = {0.06, 0.025};
  random_data(NClassA, 2, mu, sig, rho, A);
  TH2D *ClassA = build_gauss("A", A, NClassA, 0.0, 1.0);
  ClassA->SetXTitle("x_{1} [arb]");
  ClassA->SetYTitle("x_{2} [arb]");
  ClassA->GetXaxis()->CenterTitle();
  ClassA->GetYaxis()->CenterTitle();
  
  TGraph *gxA = make_plot(A, "training_A", "x_{1} [arb]", "x_{2} [arb]",
  			  34, kBlack, 1.25);

  // Class B:
  int NClassB = 100;
  double rhoB = 0.1;
  double muB[]  = {0.40, 0.35};
  double sigB[] = {0.07, 0.07};
  random_data(NClassB, 2, muB, sigB, rhoB, B);
  TH2D *ClassB = build_gauss("B", B, NClassB, 0.0, 1.0);
  TGraph *gxB = make_plot(B, "training_B", "x_{1} [arb]", "x_{2} [arb]",
  			  22, kBlue+2, 1.25);

  // Class C:
  int NClassC = 150;
  double rhoC = -0.5;
  double muC[]  = {0.45, 0.9};
  double sigC[] = {0.1, 0.03};
  random_data(NClassC, 2, muC, sigC, rhoC, C);
  TH2D *ClassC = build_gauss("C", C, NClassC, 0.0, 1.0);
  TGraph *gxC = make_plot(C, "training_C", "x_{1} [arb]", "x_{2} [arb]",
  			  33, kGreen+3, 1.75);

  // Need correct probabilities of class contribution
  double Ntotal = NClassA + NClassB + NClassC;
  double PA = NClassA / Ntotal;
  double PB = NClassB / Ntotal;
  double PC = NClassC / Ntotal;

  ////////////////////////////////////////////////////////////////////////////////
  // Calculate the probabilites that new data belongs to one of the  
  // three classes - A, B, or C!
  //
  // need an appropriate data point, let's sample one from each gaussian.
  std::vector<TVectorD> pred;
  random_data(1, 2, mu, sig, rho, pred);
  random_data(1, 2, muB, sigB, rhoB, pred);
  //random_data(1, 2, muC, sigC, rhoC, pred);
  TVectorD user_input(2);
  user_input(0) = 0.38;
  user_input(1) = 0.52;
  pred.push_back(user_input);
  
  std::map<const char*,std::vector<const char*> > results;
  
  for(unsigned int i=0; i<pred.size(); i++){
    int binxA = ClassA->GetXaxis()->FindBin(pred[i](0));
    int binyA = ClassA->GetXaxis()->FindBin(pred[i](1));
    int binxB = ClassB->GetXaxis()->FindBin(pred[i](0));
    int binyB = ClassB->GetXaxis()->FindBin(pred[i](1));
    int binxC = ClassC->GetXaxis()->FindBin(pred[i](0));
    int binyC = ClassC->GetXaxis()->FindBin(pred[i](1));
    double Pa = ClassA->GetBinContent(binxA,binyA) * PA;
    double Pb = ClassB->GetBinContent(binxB,binyB) * PB;
    double Pc = ClassC->GetBinContent(binxC,binyC) * PC;
    double norm = Pa+Pb+Pc;
    //printf("%d %d, %d %d, %d %d \n", binxA, binyA, binxB, binyB, binxC, binyC);
    printf("(x1,x2) = (%1.3f, %1.3f)\n", pred[i](0), pred[i](1));
    printf("marginal  = %1.4f \n", norm);
    double ProbA = 100.0*Pa/norm;
    double ProbB = 100.0*Pb/norm;
    double ProbC = 100.0*Pc/norm;
    printf("Prior A = %1.4f \n", PA);
    printf("Prior B = %1.4f \n", PB);
    printf("Prior C = %1.4f \n", PC);
    printf("Likelihood_A = %1.4f \n", ClassA->GetBinContent(binxA,binyA));
    printf("Likelihood_B = %1.4f \n", ClassB->GetBinContent(binxB,binyB));
    printf("Likelihood_C = %1.4f \n", ClassC->GetBinContent(binxC,binyC));
    printf("Posterior_A  = %1.4f \n", ProbA/100.);
    printf("Posterior_B  = %1.4f \n", ProbB/100.);
    printf("Posterior_C  = %1.4f \n", ProbC/100.);
    std::cout << "\n";
    results["A"].push_back(Form("%2.2f",ProbA));
    results["B"].push_back(Form("%2.2f",ProbB));
    results["C"].push_back(Form("%2.2f",ProbC));
  }

  //
  // Graph the points to classify
  TGraph *g_pred = make_plot(pred, "prediction", "x_{1} [arb]", "x_{2} [arb]",
			     20, kRed, 1.5);

  ////////////////////////////////////////////////////////////////////////////////
  // The remainder is for plotting, no need to make a ton of functions
  //   
  TCanvas *c1 = new TCanvas("c1","", 1000, 800);
  ClassA->Draw("cont1");
  gxA->Draw("Psame");
  gxA->GetXaxis()->SetLimits(0.0,1.0);
  gxA->GetHistogram()->SetMinimum(0.0);
  gxA->GetHistogram()->SetMaximum(1.0);

  ClassB->Draw("cont1same");
  gxB->Draw("Psame");

  ClassC->Draw("cont1same");
  gxC->Draw("Psame");

  g_pred->Draw("Psame");

  // make standard tlegend
  TLegend *leg = new TLegend(0.55, 0.12, 0.89, 0.35);
  leg->SetLineColor(kWhite);
  leg->AddEntry(gxA, "Class A", "p");
  leg->AddEntry(gxB, "Class B", "p");
  leg->AddEntry(gxC, "Class C", "p");
  leg->AddEntry(g_pred,"New data to classify","p");
  leg->Draw("same");

   TLegend *legp = new TLegend(0.5,0.6,0.89,0.89);
  if(ADD_LABELS){
    // visualize the probabilites
    legp->SetLineColor(kWhite);
    legp->SetTextAlign(32);
    legp->SetNColumns(4);
    legp->AddEntry((TObject*)0,"Prob. (%)","");
    legp->AddEntry((TObject*)0,"1","");
    legp->AddEntry((TObject*)0,"2","");
    legp->AddEntry((TObject*)0,"3","");
    legp->AddEntry((TObject*)0,"P_{A}","");
    for(unsigned int i=0; i<results["A"].size(); i++){
      legp->AddEntry((TObject*)0,results["A"][i],"");
    }
    legp->AddEntry((TObject*)0,"P_{B}","");
    for(unsigned int i=0; i<results["B"].size(); i++){
      legp->AddEntry((TObject*)0,results["B"][i],"");
    }
    legp->AddEntry((TObject*)0,"P_{C}","");
    for(unsigned int i=0; i<results["C"].size(); i++){
      legp->AddEntry((TObject*)0,results["C"][i],"");
    }
    legp->Draw("same");

    TArrow *a1 = new TArrow(0.1,0.4,pred[0](0), pred[0](1), 0.01, "|>");
    a1->SetLineWidth(2);
    a1->SetLineColor(kBlack);
    a1->Draw();

    TArrow *a2 = new TArrow(0.35,0.13,pred[1](0), pred[1](1), 0.01, "|>");
    a2->SetLineWidth(2);
    a2->SetLineColor(kBlack);
    a2->Draw();

    TArrow *a3 = new TArrow(0.2,0.85,pred[2](0), pred[2](1), 0.01, "|>");
    a3->SetLineWidth(2);
    a3->SetLineColor(kBlack);
    a3->Draw();

    TLatex tex;
    tex.SetTextAlign(12);
    double hsize = 0.03/2.0;
    tex.SetTextSize(2.0*hsize);
    tex.DrawLatex(0.10-hsize, 0.4-hsize,   "1");
    tex.DrawLatex(0.35, 0.13-hsize, "2");
    tex.DrawLatex(0.20-hsize, 0.85+hsize,  "3");
  }
  ////////////////////////////////////////////////////////////////////////////////
  // Lets scan through the range in both features, and make a TH2D displaying
  // the maximum probability.
  int bins = 150;
  double dx = 1.0 / double(bins);
  TH2D* hmax_prob = new TH2D("hmax_prob", "", bins, 0.0, 1.0, bins, 0.0, 1.0);
  hmax_prob->SetXTitle("x_{1} [arb]");
  hmax_prob->SetYTitle("x_{2} [arb]");
  hmax_prob->GetXaxis()->CenterTitle();
  hmax_prob->GetYaxis()->CenterTitle();

  const char* classes[] = {"A","B","C"};
  std::vector<TH2D*> class_hists;
  for(int i=0; i<3; i++){
    TH2D* hist = new TH2D(Form("hist_class%s",classes[i]), "",
				bins, 0.0, 1.0, bins, 0.0, 1.0);
    hist->SetXTitle("x_{1} [arb]");
    hist->SetYTitle("x_{2} [arb]");
    hist->GetXaxis()->CenterTitle();
    hist->GetYaxis()->CenterTitle();
    class_hists.push_back(hist);
  }
  
  // And now let's scan through [0.0, 1.0] in x1 and x2 to get the function
  double dP = 1.0 / 3.0;
  for(int xi=0; xi<bins; xi++){
    for(int yi=0; yi<bins; yi++){
      double xtemp = xi*dx + dx/2.0;
      double ytemp = yi*dx + dx/2.0;
      int binxA = ClassA->GetXaxis()->FindBin(xtemp);
      int binyA = ClassA->GetXaxis()->FindBin(ytemp);
      int binxB = ClassB->GetXaxis()->FindBin(xtemp);
      int binyB = ClassB->GetXaxis()->FindBin(ytemp);
      int binxC = ClassC->GetXaxis()->FindBin(xtemp);
      int binyC = ClassC->GetXaxis()->FindBin(ytemp);
      double Pa = ClassA->GetBinContent(binxA,binyA) * PA;
      double Pb = ClassB->GetBinContent(binxB,binyB) * PB;
      double Pc = ClassC->GetBinContent(binxC,binyC) * PC;
      double norm = Pa+Pb+Pc;
      double ProbA = Pa/norm;
      double ProbB = Pb/norm;
      double ProbC = Pc/norm;
      if( ProbA > ProbB && ProbA > ProbC ){
	double color = ProbA * dP;
	hmax_prob->SetBinContent(xi,yi, color);
	class_hists[0]->SetBinContent(xi,yi,ProbA);
      } else if(ProbB > ProbA && ProbB > ProbC){
	double color = ProbB * dP + dP;
	hmax_prob->SetBinContent(xi,yi, color);
	class_hists[1]->SetBinContent(xi,yi,ProbB);
      } else if(ProbC > ProbA && ProbC > ProbB){
	double color = ProbC * dP + 2.0*dP;
	hmax_prob->SetBinContent(xi,yi, color);
	class_hists[2]->SetBinContent(xi,yi,ProbC);
      } else{
	std::cout << "uh ohh" << std::endl;
      } 
    }
  }
  TCanvas *c3 = new TCanvas("c3","",1000,800);
  c3->cd(1);
  hmax_prob->Draw("colz");
  
  TCanvas *c4 = new TCanvas("c4","",1000,800);
  c4->Divide(2,2);
  c4->cd(1);
  ClassA->Draw("cont1");
  gxA->Draw("Psame");
  gxA->GetXaxis()->SetLimits(0.0,1.0);
  gxA->GetHistogram()->SetMinimum(0.0);
  gxA->GetHistogram()->SetMaximum(1.0);
  ClassB->Draw("cont1same");
  gxB->Draw("Psame");
  ClassC->Draw("cont1same");
  gxC->Draw("Psame");
  g_pred->Draw("Psame");
  leg->Draw("same");
  legp->Draw("same");
  for( int i=0; i<3; i++ ){
    c4->cd(i+2);
    class_hists[i]->Draw("colz");
  }
  
  return;
}
