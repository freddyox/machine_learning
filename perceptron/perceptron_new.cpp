#include "TFile.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TRandom.h"
#include "TAxis.h"
#include "TVectorD.h"
#include "TF1.h"
#include "TH1D.h"

#include <iostream>
#include <vector>

#define HELP 0

////////////////////////////////////////////////////////////////////////////////
// Perceptron - uses K-fold cross validation to generate a linear
//              decision boundary for linearly separable data.
//

// Used in several functions, need global scope
TRandom fRand(0);

// Easy way to generate a TGraph
TGraph* make_plot(int N, double* x, double* y, const char* name,
		  const char* xaxis, const char* yaxis,
		  int marker, int color, float markersize, const char* title);

// Fill arrays and smear them according to pars and weights
void fill_arrays(int Ndata, int nvars, double* pars, double* weights,
		 double* x1, double* y1, double* x2, double* y2){
  // Generate Data:
  for(int i=0; i<Ndata; i++){
    // reset
    double temppars[nvars];
    for(int nv=0; nv<nvars; nv++) temppars[nv] = pars[nv];
    
    // smear:
    for(int nv=0; nv<nvars; nv++){
      temppars[nv] += fRand.Gaus( 0.0, pars[nv] * weights[nv] );
    }
    
    // fill
    x1[i] = temppars[0];
    y1[i] = temppars[1];
    x2[i] = temppars[2];
    y2[i] = temppars[3];
  }
}

// Returns weight vector
std::vector<double> train(unsigned int, double*, double*, int*, double, int);

// Returns weight vector
std::vector<double> train_Kfold(unsigned int N, std::vector<double> X, std::vector<double> Y,
				std::vector<int> H, double eta, int NAttempts, unsigned int Kfold);

////////////////////////////////////////////////////////////////////////////////
// Driver script - all in one function as this is a development script
//
void perceptron_new(int Ndata=25){
  double norm = 25.0;
  double norm_weight = 3.5;
  double mass1 = 10.0/norm;
  double radius1 = 10.0/norm;
  double mass2 = 25.0/norm;
  double radius2 = 14.0/norm;

  // Need normalized data
  double weights[4] = {0.55, 0.5, 0.6, 0.45};
  for(int i=0; i<4; i++) weights[i] /= norm_weight;
  double vars[4] = {mass1,radius1,mass2,radius2};
  int nvars = sizeof(weights) / sizeof*(weights);
  double x1_train[Ndata], y1_train[Ndata];
  double x2_train[Ndata], y2_train[Ndata];

  fill_arrays(Ndata,nvars,vars,weights,x1_train,y1_train,x2_train,y2_train);

  ////////////////////////////////////////////////////////////////////////////////
  // THE TRAINING DATA
  //
  std::vector<double> X,Y;
  std::vector<int> H;
  for(int i=0; i<Ndata; i++){
    X.push_back( x1_train[i] );
    H.push_back( -1 );
    X.push_back( x2_train[i] );
    H.push_back( 1 );
    Y.push_back( y1_train[i] );
    Y.push_back( y2_train[i] );
  }
  double eta = 0.2;
  unsigned int Xsize = X.size();
  std::vector<double> W = train(Xsize, &(X[0]), &(Y[0]), &(H[0]), eta, 10);
  std::vector<double> W_Kfold = train_Kfold(Xsize, X, Y, H, eta, 10, Xsize);
  
  printf("Initial W: %f \t %f \t %f \n", W[3], W[4], W[5] );
  printf("  Final W: %f \t %f \t %f \n", W[0], W[1], W[2] );

  if( fabs(W[2]) < 1e-6 ) {std::cout << "W[2] is really small!" << std::endl;}
  
  double m = -W[1] / W[2];
  double b = -W[0] / W[2];
  
  TF1* fClassifier = new TF1("fClassifier","pol1",-1.0,2.0);
  fClassifier->SetParameter(0,b);
  fClassifier->SetParameter(1,m);
  fClassifier->SetLineWidth(2);
  fClassifier->SetLineStyle(9);
  fClassifier->SetLineColor(kBlack);

  double mfold = -W_Kfold[1] / W_Kfold[2];
  double bfold = -W_Kfold[0] / W_Kfold[2];
  
  TF1* fClassifier_K = new TF1("fClassifier_K","pol1",-1.0,2.0);
  fClassifier_K->SetParameter(0,bfold);
  fClassifier_K->SetParameter(1,mfold);
  fClassifier_K->SetLineWidth(5);
  fClassifier_K->SetLineColor(kBlack);
  
  ////////////////////////////////////////////////////////////////////////////////
  // Plot Everything for an easy visual:

  TGraph *gtemp_t1 = make_plot(Ndata, x1_train, y1_train, "Mass1_radius1",
			       "Mass [arb]", "Radius [arb]", 20, kBlue, 1.0, "Train");
  TGraph *gtemp_t2 = make_plot(Ndata, x2_train, y2_train, "Mass2_radius2",
			       "Mass [arb]", "Radius [arb]", 23, kRed, 1.0, "Train");			     

  TMultiGraph *mg = new TMultiGraph();
  mg->Add(gtemp_t1);
  mg->Add(gtemp_t2);

  ////////////////////////////////////////////////////////////////////////////////
  // GENERATE ANOTHER SET OF DATA AND TEST:
  // Now let's test it:
  int Npred = 200;
  double x1_pred[Npred], y1_pred[Npred];
  double x2_pred[Npred], y2_pred[Npred];
  
  fill_arrays(Npred,nvars,vars,weights,x1_pred,y1_pred,x2_pred,y2_pred);
  
  std::vector<double> Xpred,Ypred;
  std::vector<int> Hpred;
  for(int i=0; i<Npred; i++){
    Xpred.push_back( x1_pred[i] );
    Hpred.push_back( -1 );
    
    Xpred.push_back( x2_pred[i] );
    Hpred.push_back( 1 );
    
    Ypred.push_back( y1_pred[i] );
    Ypred.push_back( y2_pred[i] );
  }

  int correct = 0;
  int incorrect = 0;
  int Ntotal = Xpred.size();
  
  for(unsigned int i=0; i<Xpred.size(); i++){
    double xi = Xpred[i];
    double yi = Ypred[i];
    int hi = Hpred[i];
    double sum = 1.0 * W[0] + xi * W[1] + yi * W[2];
    int predicted = 0;
    if( sum > 0.0 ) predicted = 1;
    if( sum < 0.0 ) predicted = -1;

    if( predicted == hi ) correct++;
    if( predicted != hi ) incorrect++;
  }
  double rcorr = (double(correct) / double(Ntotal)) * 100.0;
  double rincorr = (double(incorrect) / double(Ntotal)) * 100.0;
  std::cout << "Total # thrown: " << Ntotal << std::endl;
  std::cout << "Correctly predicted = " << rcorr << "%" << std::endl;
  std::cout << "Incorrectly pred    = " << rincorr << "%" << std::endl;

  ////////////////////////////////////////////////////////////////////////////////
  // Draw everything for visualization
  //
  TGraph *gtemp_p1 = make_plot(Npred, x1_pred, y1_pred, "Mass1_radius1_pred",
			       "Mass [arb]","Radius [arb]", 20, kBlue, 1.0, "Prediction");

  TGraph *gtemp_p2 = make_plot(Npred, x2_pred, y2_pred, "Mass2_radius2_pred",
			       "Mass [arb]","Radius [arb]", 23, kRed, 1.0, "Prediction");

  TMultiGraph *mg1 = new TMultiGraph();
  mg1->Add(gtemp_p1);
  mg1->Add(gtemp_p2);

  TCanvas *c1 = new TCanvas("c1","",1000,600);
  c1->Divide(2,1);
  c1->cd(1);
  mg->Draw("AP");
  mg->GetXaxis()->SetTitle("Mass [arb]");
  mg->GetYaxis()->SetTitle("Radius [arb]");
  mg->GetXaxis()->CenterTitle();
  mg->GetYaxis()->CenterTitle();
  mg->GetXaxis()->SetLimits(0.0,1.4);
  mg->GetHistogram()->SetMinimum(0.0);
  mg->GetHistogram()->SetMaximum(1.0);
  fClassifier->Draw("same");
  fClassifier_K->Draw("same");
  TLegend *leg = new TLegend(0.6,0.6,0.89,0.89);
  leg->SetLineColor(kWhite);
  leg->AddEntry(fClassifier,"Naive","l");
  leg->AddEntry(fClassifier_K,"K-fold","l");
  leg->Draw("same");
  
  c1->cd(2);
  mg1->Draw("AP");
  mg1->GetXaxis()->SetTitle("Mass [arb]");
  mg1->GetYaxis()->SetTitle("Radius [arb]");
  mg1->GetXaxis()->CenterTitle();
  mg1->GetYaxis()->CenterTitle();
  mg1->GetXaxis()->SetLimits(0.0,1.4);
  mg1->GetHistogram()->SetMinimum(0.0);
  mg1->GetHistogram()->SetMaximum(1.0);
  fClassifier_K->Draw("same");
  return;
}

////////////////////////////////////////////////////////////////////////////////
// Generate a scatter plot
//
TGraph* make_plot(int N, double* x, double* y,  const char* name,
		  const char* xaxis, const char* yaxis,
		  int marker, int color, float markersize, const char* title){
  TGraph *gtemp = new TGraph(N,x,y);
  gtemp->SetTitle(title);
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
// Naive training routine - immediately stops when decision boundary is found
//
std::vector<double> train(unsigned int N, double* X, double* Y, int* H, double eta, int NAttempts){

  int dimension = 2;
  std::vector<double> W;
  double Winitial[dimension+1]; 
  for(int i=0; i<=dimension; i++){
    double val = fRand.Rndm();
    W.push_back( val );
    Winitial[i] = val;
  }

  int attempt = 1;

  while( attempt <= NAttempts ){
    attempt++;
    
    // get a coordinate:
    int do_we_continue = 0;
    for(unsigned int index=0; index < N; index++){
      double xi = X[index];
      double yi = Y[index];
      int hi = H[index];

      // do the summation:
      // sum w^T x = w0*1 + w1*x + w2*y
      int checking = 0;
      double sum = 1.0 * W[0] + xi * W[1] + yi * W[2];
      if( (sum > 0.0 && hi < 0) ||
	  (sum < 0.0 && hi > 0) ){
	W[0] += eta * hi * 1.0;
	W[1] += eta * hi * xi;
	W[2] += eta * hi * yi;
	checking = 1;
      } 
    }
  }
  // indices 3,4,5 are the initial random values of W
  // 0,1,2 are the final
  for(int i=0; i<=dimension; i++) W.push_back( Winitial[i] );

  return W;
}

////////////////////////////////////////////////////////////////////////////////
// K-fold cross-validation routine - runs the naive training routine many
// times for slices of the training data, and the decision boundary is then
// taken as the average yielding much better results
//
std::vector<double> train_Kfold(unsigned int N, std::vector<double> X, std::vector<double> Y,
				std::vector<int> H, double eta, int NAttempts, unsigned int Kfold){
  // Need to ignore an index systematically
  if( Kfold > N ) Kfold = N;

  std::vector<double> W_averaged;
  for(int i=0; i<3; i++) {
    W_averaged.push_back( 0.0);
  }
  
  for(unsigned int i=0; i<Kfold; i++){
    std::vector<double> Xnew = X;
    std::vector<double> Ynew = Y;
    std::vector<int> Hnew = H;
    Xnew.erase(Xnew.begin()+i);
    Ynew.erase(Ynew.begin()+i);
    Hnew.erase(Hnew.begin()+i);
    std::vector<double> W = train(Xnew.size(), &(Xnew[0]), &(Ynew[0]), &(Hnew[0]), eta, 10);

    for(int j=0; j<3; j++){
      W_averaged[j] += W[j];
    }
  }
  return W_averaged;
}
