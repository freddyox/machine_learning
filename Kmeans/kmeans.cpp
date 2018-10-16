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

#include <iostream>
#include <vector>
#include <map>

#define DRAW_EVO 1 // Draw the large algorithmic evolution canvas
#define MAKE_GIF 0 // Note that this continues to add data to animation.gif

////////////////////////////////////////////////////////////////////////////////
//             K-means clustering algorithm (unsupervised)
//                      Obrecht - 10/15/2018
//
//  Note: I use TVectorD as a pseudo-structure for an (x1,x2,K) point, i.e. a
//  vector<TVectorD> is a STL vector of data points where each element has an
//  x1,x2 coordinate and an assigned cluster K (simply a book-keeping device).
//  Clusters are randomly initiated, and points are added based on minimizing
//  the distance in this x1,x2 feature-space.
//

////////////////////////////////////////////////////////////////////////////////
// Simple method to make TGraphs - returns pointer with all assigned properties
//
TGraph* make_plot(std::vector<TVectorD> &data,  const char* name,
		  const char* xaxis, const char* yaxis,
		  int marker, int color, float markersize){
  int N = data.size();
  double x[N],y[N];
  for(int i=0; i<N; i++){
    x[i] = data[i](0);     // TGraph likes arrays
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
//  The routine fill the STL vector data
//
void random_data(int N, int nvars, double *means, double *sig, double rho,
		 std::vector<TVectorD>& data){

  TRandom fRand(0);
  
  // Local variables for human-readability
  double mu1 = means[0];  // mean in x1
  double mu2 = means[1];  // mean in x2
  double sig1 = sig[0];   // std deviation in x1
  double sig2 = sig[1];   //   " "
  for(int i=0; i<N; i++){
    double z1 = fRand.Gaus(mu1, sig1); // normally independent in x1 direction
    double z2 = fRand.Gaus(mu2, sig2); // normally independent in x2 direction
    double x1_normal = z1; 
    double x2_normal = rho*z1 + sqrt(1.0 - rho*rho) * z2; // add x1/x2 correlation
    TVectorD pt(3);
    pt(0) = x1_normal;
    pt(1) = x2_normal;  
    pt(2) = 0;           // initialize the cluster assignment to 0
    data.push_back(pt);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Utility function for printing vector<TVectorD>
//
void print(std::vector<TVectorD>& data){
  for(unsigned int i=0; i<data.size(); i++)
    printf("(%1.3f, %1.3f, %1.0f), ", data[i](0), data[i](1), data[i](2));
  std::cout << "\b\b\b" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Partition the data based on the identified cluster; therefore, this routine
// fills a vector of graphs of length K (or # of clusters).
//
void get_cluster_graphs(std::vector<TVectorD> &data, std::vector<TGraph*> &output){
  // key is cluster number, val is data
  std::map<int, std::vector<TVectorD> > cluster_data; 
  for(unsigned int i=0; i<data.size(); i++){
    int K = int(data[i](2));
    cluster_data[K].push_back( data[i] );
  }

  // If we have more than 10 clusters, this will start repeating itself...
  const char* names[] = {"A","B","C","D","E",
			 "F","G","H","I","J"};
  int markers[] = {20, 21, 22, 23, 34,
		   25, 26, 27, 28, 29};
  int colors[]  = {kRed, kBlue, kGreen-3, kViolet, kOrange+9,
		   kCyan-2, kOrange+9, kMagenta+3, kGray+2, kAzure+10};
  int Nnames = sizeof(names) / sizeof*(names);
  
  std::map<int, std::vector<TVectorD> >::iterator mit;
  for(mit=cluster_data.begin(); mit!=cluster_data.end(); mit++){
    int K = mit->first % Nnames; // protection against seg-fault
    TGraph *gtemp = make_plot(cluster_data[K], names[K], "x_{1} [arb]", "x_{2} [arb]",
			      markers[K], colors[K], 1.3);
    output.push_back(gtemp);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Follow the evolution of the algorithm, need to store many plots. Since this
// is a simple script, we need to declare global containers which is fine for now
//
std::map<int, std::vector<TGraph*> > Cluster_Evo; // key is algo iteration #
std::map<int, TGraph*> KMean_Evo;                 // key is algo iteration #
void fill_cluster_evo(std::vector<TVectorD> &data, int niter){
  std::vector<TGraph*> htemp;
  get_cluster_graphs(data, htemp);
  for(unsigned int i=0; i<htemp.size(); i++)
    Cluster_Evo[niter].push_back(htemp[i]);
}
//
void fill_mean_evo(std::vector<TVectorD> &mu, int niter){
  TGraph *gtemp = make_plot(mu, Form("Kmeans_%d",niter),  "x_{1} [arb]", "x_{2} [arb]",
			    29, kBlack, 2.0);
  KMean_Evo[niter] = gtemp;
}

////////////////////////////////////////////////////////////////////////////////
// Assign mu-vector initial values using randomization and the data for bounds
//
void rndm_init(std::vector<TVectorD>& mu, int K, std::vector<TVectorD> &data){
  // If the random number is outside of the dataset, it doesn't work well.
  // Therefore, let's use the data to define a sub-region for random initialization
  TVectorD min(2),max(2),diff(2);
  min(0) = 1.e6;  min(1) = 1.e6;
  max(0) = -1.e6; max(1) = -1.e6;
  for(unsigned int i=0; i<data.size(); i++){
    for(int v=0; v<2; v++){
      if( data[i](v) < min(v) ) min(v) = data[i](v);
      if( data[i](v) > max(v) ) max(v) = data[i](v);
    }
  }

  diff(0) = max(0)-min(0);
  diff(1) = max(1)-min(1);
  
  TRandom fRand(0);
  for(int i=0; i<K; i++){
    TVectorD temppt(3);  
    for(int j=0; j<2; j++) temppt(j) = (fRand.Rndm()*diff(j) + min(j));
    temppt(2) = i;
    mu.push_back( temppt );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Assign data to a cluster randomly - requires total # of clusters K which
//  is why this method has to wait until K is specified.
//
void rndm_init_clust(std::vector<TVectorD>& data, int K){
  TRandom fRand(0);
  for(unsigned int i=0; i<data.size(); i++){
    int rnd_K = fRand.Integer(K+1); // [ 0, imax-1 ] => K+1 is correct
    data[i](2) = double(rnd_K);
  }
}

////////////////////////////////////////////////////////////////////////////////
// The update rule for the mu-vector. We need to re-calculate the mean for
// each cluster using simple average (but I need to find which data pt belongs
// to which cluster for this to work...)
//
void update_means(std::vector<TVectorD>& data, std::vector<TVectorD>& mu){
  // keys are cluster #
  std::map<int, int> K_N;         // count # of objects in cluster K
  std::map<int, double> K_x, K_y; // Keep track of position

  // intialize the sums correctly
  for(unsigned i=0; i<mu.size(); i++) {
    K_N[int(mu[i](2))] = 0;
    K_x[int(mu[i](2))] = 0.0;
    K_y[int(mu[i](2))] = 0.0;
  }
  // Do the sums
  for(unsigned i=0; i<data.size(); i++){
    int K = int(data[i](2));
    K_N[K]++;
    K_x[K] += data[i](0);
    K_y[K] += data[i](1);
  }
  // Perform the calculation if a data points is found (avoid NaN)
  for(unsigned i=0; i<mu.size(); i++){
    int K = int(mu[i](2));
    if(K_N[K]<=0) continue; // don't update, nothing was found
    double mu_x_new = K_x[K] / K_N[K];
    double mu_y_new = K_y[K] / K_N[K];
    TVectorD pt(3);
    pt(0) = mu_x_new;
    pt(1) = mu_y_new;
    pt(2) = K;
    mu[i] = pt;
  }
}

////////////////////////////////////////////////////////////////////////////////
// K-means clustering - the main algorithm routine
//  input:  1) data, 2) # of clusters to look for, and 3) # of iteration attempts 
//  output: Vector holding the mean of each cluster      
void cluster(std::vector<TVectorD> &data, int K, int Niterations,
	     std::vector<TVectorD>& Kmeans){
  std::vector<TVectorD> clust_mu;
  rndm_init(clust_mu, K, data);    // initialize the means of K clusters
  rndm_init_clust(data,K);         // assign each data point a cluster
  std::cout << "Initial Kmeans: ";
  print(clust_mu);
  
  int niter = 0; 
  while( niter < Niterations ){
    niter++;                         // keep track of iteration #
    int converged = true;            // keep track of updates and when to break
      
    // Let's examine the data:
    for(unsigned int i=0; i<data.size(); i++){
      TVectorD object = data[i];     // For readibility
      int object_k = int(object(2)); // Get the cluster #
      
      int min_k = -1;                // Assign object to this cluster
      double min_distance = 1.0e6;   // Initialize to something large

      // Assign object to a cluster based off the min distance
      // between cluster_k's mean and the object position
      for(unsigned int k=0; k<clust_mu.size(); k++){
	TVectorD mu = clust_mu[k];    // cluster's mean position
	int kclust = int(mu(2));      // identify the mu cluster
	TVectorD delta = object - mu; // convenience
	double D = sqrt( pow(delta(0),2.0) + pow(delta(1),2.0) );
	if( D < min_distance ){
	  min_distance = D;
	  min_k = kclust;
	}
      }
      // Protection if min_k not assigned, which I think is impossible but OK
      if( min_k < 0 ) {
	std::cout << "UHHHH OHHHHHH CLUSTER NOT ASSIGNED!!" << std::endl;
	continue;
      }
      ////////////////////////////////////////
      // If we get here, we've assigned the object to a cluster. Check if we are done.
      if( object_k != min_k ) converged = false;
      
      // Update the object's cluster K if we need to	
      if( !converged ) data[i](2) = double(min_k);
    }

    // Visually keep track of what is going on to make pretty plots
    fill_cluster_evo(data,niter);
    fill_mean_evo(clust_mu,niter);
    
    // If we haven't converged yet, then we need to update the K-means,
    // otherwise let's exit the loop as we are done.
    std::cout << "Iteration " << niter << std::endl;
    std::cout << "Converged " << Form("%s",converged?"yes":"no") << std::endl;
    if( !converged ){
      std::cout << "before: ";
      print(clust_mu);
      update_means(data, clust_mu);
      std::cout << "after:  ";
      print(clust_mu);
    } else {
      break; // exit
    }
  }
  Kmeans = clust_mu; // final position of the means 
}

////////////////////////////////////////////////////////////////////////////////
// Main driver script
//
void kmeans(){
  // Some style commands...
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetOptStat(0);
  
  // Need to generate some data to look at...
  std::vector<TVectorD> data;
  double muA[]  = {0.30, 0.25};
  double sigA[] = {0.05, 0.05};
  double muB[]  = {0.60, 0.34};
  double sigB[] = {0.06, 0.05};
  double muC[]  = {0.40, 0.60};
  double sigC[] = {0.05, 0.07};
  double muD[]  = {0.20, 0.70};
  double sigD[] = {0.05, 0.07};
  double muE[]  = {0.68, 0.68};
  double sigE[] = {0.05, 0.07};
  random_data(50, 2, muA, sigA, 0.0, data);
  random_data(60, 2, muB, sigB, 0.0, data);
  random_data(100, 2, muC, sigC, 0.0, data);
  random_data(100, 2, muD, sigD, 0.0, data);
  random_data(100, 2, muE, sigE, 0.0, data);
  //
  int K = 5;
  int N = 100;
  std::vector<TVectorD> Kmeans;
  cluster(data,K,N,Kmeans);

  // Make a plot before the algorithm is applied
  TGraph *gdata = make_plot(data, "unlabeled_data", "x_{1} [arb]", "x_{2} [arb]",
			    20, kBlack, 1.3);

  // Make a plot after the algorithm
  std::vector<TGraph*> clusters;
  get_cluster_graphs(data, clusters);

  // Need to visualize the cluster means (hence K-means)
  TGraph *gKmeans = make_plot(Kmeans, "K-means",  "x_{1} [arb]", "x_{2} [arb]",
			      29, kBlack, 2.0);
  
  // Trick to make TCanvas look nice...
  TH1D *hframe = new TH1D("hframe","",100,0.0,1.0);
  hframe->GetYaxis()->SetRangeUser(0.0,1.0);
  hframe->SetTitleFont(63,"XYZ");
  hframe->SetLabelFont(63,"XYZ");
  hframe->SetTitleSize(30,"XYZ");
  hframe->SetLabelSize(30,"XYZ");
  hframe->SetTitleOffset(1.3,"XY");  
  hframe->SetNdivisions(505,"XYZ");
  hframe->GetXaxis()->SetTitle("x_{1} [arb]");
  hframe->GetYaxis()->SetTitle("x_{2} [arb]");
  hframe->GetXaxis()->CenterTitle();
  hframe->GetYaxis()->CenterTitle();

  // And plot everything...
  TCanvas *c1 = new TCanvas("c1","",1200,900);
  c1->Divide(2,1);
  c1->cd(1);
  gPad->SetLeftMargin(-0.2);
  gPad->SetBottomMargin(-0.14);
  hframe->Draw();
  gdata->Draw("Psame");

  c1->cd(2);
  hframe->Draw();
  const char* names[] = {"A","B","C","D","E",
			 "F","G","H","I","J"};
  TLegend *leg = new TLegend(0.5,0.66,0.89,0.89);
  leg->SetLineColor(kWhite);
  leg->SetTextFont(63);
  leg->SetNColumns(2);
  
  for(unsigned int i=0; i<clusters.size(); i++){
    clusters[i]->Draw("Psame");
    leg->AddEntry(clusters[i],Form("Class %s",names[i]), "p");
  }
  gKmeans->Draw("Psame");
  leg->AddEntry(gKmeans, "K-Means", "p");
  
  leg->Draw("same");
  
  ////////////////////////////////////////////////////////////////////////////////
  // Drawing the evolution of the algorithm
  //
  if( DRAW_EVO ){
    TH1D *hframe_evo = new TH1D("hframe_evo","",100,0.0,1.0);
    hframe_evo->GetYaxis()->SetRangeUser(0.0,1.0);
    hframe_evo->SetTitleFont(63,"XYZ");
    hframe_evo->SetLabelFont(63,"XYZ");
    hframe_evo->SetTitleSize(13,"XYZ");
    hframe_evo->SetLabelSize(13,"XYZ");
    hframe_evo->SetTitleOffset(2.5,"XY");  
    hframe_evo->SetNdivisions(505,"XYZ");
    hframe_evo->GetXaxis()->SetTitle("x_{1} [arb]");
    hframe_evo->GetYaxis()->SetTitle("x_{2} [arb]");
    hframe_evo->GetXaxis()->CenterTitle();
    hframe_evo->GetYaxis()->CenterTitle();

    TCanvas *c2 = new TCanvas("c2","",1200,900);
    int Ncells = Cluster_Evo.size() + 1; // +1 for the unlabeled data
    int divx = 3, divy=1;                //
    while( Ncells > divx*divy ) divy++;  // grow if we need to
    c2->Divide(divx, divy, -1, -1);      // Draw a lattice of plots

    // First cell is unlabeled data
    c2->cd(1);
    hframe_evo->Draw();
    gdata->Draw("Psame");

    // Other cells are algorithmic evolution
    int canv_count = 2;
    std::map<int, std::vector<TGraph*> >::iterator mivt;
    for(mivt=Cluster_Evo.begin(); mivt!=Cluster_Evo.end(); mivt++){
      c2->cd(canv_count++);
      hframe_evo->Draw();
      for(unsigned int k=0; k<(mivt->second).size(); k++){
	(mivt->second)[k]->Draw("Psame");
      }
      KMean_Evo[mivt->first]->Draw("Psame");
    }

    // Remaining cells are simply empty.
    while( canv_count <= divx*divy){
      c2->cd(canv_count++);
      hframe_evo->Draw();
    }
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  // Generate an animation
  if( MAKE_GIF ){
    TH1D *hframegif = new TH1D("hframegif","",100,0.0,1.0);
    hframegif->GetYaxis()->SetRangeUser(0.0,1.0);
    hframegif->SetTitleFont(63,"XYZ");
    hframegif->SetLabelFont(63,"XYZ");
    hframegif->SetTitleSize(30,"XYZ");
    hframegif->SetLabelSize(30,"XYZ");
    hframegif->SetTitleOffset(1.1,"XY");  
    hframegif->SetNdivisions(505,"XYZ");
    hframegif->GetXaxis()->SetTitle("x_{1} [arb]");
    hframegif->GetYaxis()->SetTitle("x_{2} [arb]");
    hframegif->GetXaxis()->CenterTitle();
    hframegif->GetYaxis()->CenterTitle();

    TCanvas *c3 = new TCanvas("c3","",800,800);
    c3->cd(1);
    hframegif->Draw();
    gdata->Draw("Psame");
    TLatex tex;
    tex.SetTextAlign(12);
    tex.SetTextSize(0.045);
    tex.DrawLatex(0.65,0.15,"Unclassified");
    c3->Print("animation.gif+99");
   
    int counter = 1;
    std::map<int, std::vector<TGraph*> >::iterator mivt;
    for(mivt=Cluster_Evo.begin(); mivt!=Cluster_Evo.end(); mivt++){
      c3->Clear();
      c3->Update();
      hframegif->Draw();
      for(unsigned int k=0; k<(mivt->second).size(); k++){
	(mivt->second)[k]->Draw("Psame");
      }
      KMean_Evo[mivt->first]->Draw("Psame");
      tex.DrawLatex(0.7,0.15,Form("Iteration %d",counter++));
      c3->Print("animation.gif+99");
    }
    c3->Close();
  }
  
  return;
}
