// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2014-2015 Tsinghua University
// Author: wb.th08@gmail.com (Bin Wang), ozj@tsinghua.edu.cn (Zhijian Ou) 
//
// All h, cpp, cc, and script files (e.g. bat, sh, pl, py) should include the above 
// license declaration. Different coding language may use different comment styles.


#ifndef _MLtrain
#include "trf-model.h"
#include "trf-sa-train.h"
#include <omp.h>
using namespace trf;

char *cfg_pathVocab = NULL;

int cfg_nFeatOrder = 2;
char *cfg_pathFeatStyle = NULL;
int cfg_nMaxLen = 0;

char *cfg_pathTrain = NULL;
char *cfg_pathValid = NULL;
char *cfg_pathTest = NULL;

char *cfg_pathModelRead = NULL;
char *cfg_pathModelWrite = NULL;

int cfg_nThread = 1;

int cfg_nIterTotalNum = 1000;
int cfg_nMiniBatch = 300;
int cfg_t0 = 500;
char *cfg_gamma_lambda = NULL;
char *cfg_gamma_zeta = NULL;
bool cfg_bUnupdateLambda = false;
bool cfg_bUnupdateZeta = false;
int cfg_nAvgBeg = 0;

float cfg_fRegL2 = 0;

bool cfg_bInitValue = false;
int cfg_nPrintPerIter = 1;
char *cfg_strWriteAtIter = NULL;

char *cfg_pathWriteMean = NULL;
char *cfg_pathWriteVar = NULL;

Option opt;

_wbMain
{
	opt.Add(wbOPT_STRING, "vocab", &cfg_pathVocab, "The vocabulary");
	opt.Add(wbOPT_STRING, "feat", &cfg_pathFeatStyle, "a feature style file. Set this value will disable -order");
	opt.Add(wbOPT_INT, "order", &cfg_nFeatOrder, "the ngram feature order");
	opt.Add(wbOPT_INT, "len", &cfg_nMaxLen, "the maximum length of TRF");
	opt.Add(wbOPT_STRING, "train", &cfg_pathTrain, "Training corpus (TXT)");
	opt.Add(wbOPT_STRING, "valid", &cfg_pathValid, "valid corpus (TXT)");
	opt.Add(wbOPT_STRING, "test", &cfg_pathTest, "test corpus (TXT)");

	opt.Add(wbOPT_STRING, "read", &cfg_pathModelRead, "Read the init model to train");
	opt.Add(wbOPT_STRING, "write", &cfg_pathModelWrite, "Output model");

	opt.Add(wbOPT_INT, "iter", &cfg_nIterTotalNum, "iter total number");
	opt.Add(wbOPT_INT, "thread", &cfg_nThread, "The thread number");
	opt.Add(wbOPT_INT, "mini-batch", &cfg_nMiniBatch, "mini-batch");
	opt.Add(wbOPT_INT, "t0", &cfg_t0, "t0");
	opt.Add(wbOPT_STRING, "gamma-lambda", &cfg_gamma_lambda, "learning rate of lambda");
	opt.Add(wbOPT_STRING, "gamma-zeta", &cfg_gamma_zeta, "learning rate of zeta");
	opt.Add(wbOPT_TRUE, "unupdate-lambda", &cfg_bUnupdateLambda, "don't update lambda");
	opt.Add(wbOPT_TRUE, "unupdate-zeta", &cfg_bUnupdateZeta, "don't update zeta");
	opt.Add(wbOPT_INT, "tavg", &cfg_nAvgBeg, ">0 then apply averaging");
	opt.Add(wbOPT_FLOAT, "L2", &cfg_fRegL2, "regularization L2");

	opt.Add(wbOPT_TRUE, "init", &cfg_bInitValue, "Re-init the parameters");
	opt.Add(wbOPT_INT, "print-per-iter", &cfg_nPrintPerIter, "print the LL per iterations");
	opt.Add(wbOPT_STRING, "write-at-iter", &cfg_strWriteAtIter, "write the LL per iteration, such as [1:100:1000]");

	opt.Add(wbOPT_STRING, "write-mean", &cfg_pathWriteMean, "write the expecataion on training set");
	opt.Add(wbOPT_STRING, "write-var", &cfg_pathWriteVar, "write the variance on training set");

	opt.Parse(_argc, _argv);

	lout << "*********************************************" << endl;
	lout << "         TRF_SAtrain.exe { by Bin Wang }        " << endl;
	lout << "\t" << __DATE__ << "\t" << __TIME__ << "\t" << endl;
	lout << "**********************************************" << endl;

	srand(time(NULL));
	omp_set_num_threads(cfg_nThread);
	lout << "[OMP] omp_thread = " << omp_get_max_threads() << endl;
	Title::SetGlobalTitle(String(cfg_pathModelWrite).FileName());

	Vocab *pv = new Vocab(cfg_pathVocab);
	Model m(pv, cfg_nMaxLen);
	if (cfg_pathModelRead) {
		m.ReadT(cfg_pathModelRead);
	}
	else {
		m.LoadFromCorpus(cfg_pathTrain, cfg_pathFeatStyle, cfg_nFeatOrder);
	}

	/**/
// 	File fexp("test.exp", "wt");
// 	File fsamp("test.samp", "wt");
// 	int nLen = 5;
// 	Seq seq(nLen);
// 	seq.Random(pv);
// 	Vec<double> exp1(m.GetParamNum()), exp2(m.GetParamNum());
// 	Vec<double> pi1(m.GetMaxLen()+1), pi2(m.GetMaxLen()+1);
// 	pi2.Fill(1.0 / m.GetMaxLen());
// 	m.SetPi(pi2.GetBuf());
// 
// 	m.ExactNormalize();
// 	lout << "Exact" << endl;
// 	m.GetNodeExp(exp2.GetBuf());
// 	fexp.PrintArray("%f ", exp2.GetBuf(), exp2.GetSize());
// 	lout << "Sample" << endl;
// 	exp1.Fill(0);
// 	pi1.Fill(0);
// 	for (int i = 1; i <= 10000; i++) {
// 		m.Sample(seq);
// 		m.FeatCount(seq, exp1.GetBuf());
// 		pi1[seq.GetLen()]++;
// 
// 		seq.Print(fsamp);
// 
// 		if (i % 100 == 0) {
// 			exp1 /= i;
// 			fexp.PrintArray("%f ", exp1.GetBuf(), exp1.GetSize());
// 			exp1 *= i;
// 		}
// 	}
// 	pi1 /= 10000;
// 	lout.output(pi1.GetBuf()+1, pi1.GetSize()-1) << endl;
// 	return 1;
	/**/

	CorpusTxt *pTrain = (cfg_pathTrain) ? new CorpusTxt(cfg_pathTrain) : NULL;
	CorpusTxt *pValid = (cfg_pathValid) ? new CorpusTxt(cfg_pathValid) : NULL;
	CorpusTxt *pTest = (cfg_pathTest) ? new CorpusTxt(cfg_pathTest) : NULL;

	SAfunc func;
	func.m_fdbg.Open(String(cfg_pathModelWrite).FileName() + ".sadbg", "wt");
	func.m_fmean.Open(cfg_pathWriteMean, "wt");
	func.m_fvar.Open(cfg_pathWriteVar, "wt");
//  	func.m_fparm.Open(String(cfg_pathModelWrite).FileName() + ".parm", "wt");
//  	func.m_fgrad.Open(String(cfg_pathModelWrite).FileName() + ".grad", "wt");
// 	func.m_fexp.Open(String(cfg_pathModelWrite).FileName() + ".expt", "wt");
//   	func.m_fsamp.Open(String(cfg_pathModelWrite).FileName() + ".samp", "wt");
//   	func.m_ftrain.Open(String(cfg_pathModelWrite).FileName() + ".train", "wt");
	func.m_pathOutputModel = cfg_pathModelWrite;
	func.m_fRegL2 = cfg_fRegL2;
	func.Reset(&m, pTrain, pValid, pTest, cfg_nMiniBatch);
	func.m_fRegL2 = cfg_fRegL2;
	func.PrintInfo();



	SAtrain solve(&func);
	solve.m_nIterMax = cfg_nIterTotalNum; // fix the iteration number
	solve.m_gain_lambda.Reset(cfg_gamma_lambda ? cfg_gamma_lambda : "0,0", cfg_t0);
	solve.m_gain_zeta.Reset(cfg_gamma_zeta ? cfg_gamma_zeta : "0,0.6", cfg_t0);
	solve.m_bUpdate_lambda = !cfg_bUnupdateLambda;
	solve.m_bUpdate_zeta = !cfg_bUnupdateZeta;
	solve.m_nAvgBeg = cfg_nAvgBeg;
	solve.m_nPrintPerIter = cfg_nPrintPerIter;
	VecUnfold(cfg_strWriteAtIter, solve.m_aWriteAtIter);
	solve.PrintInfo();


	/* set initial values */
	bool bInitWeight = (!cfg_pathModelRead) || (cfg_bInitValue && !cfg_bUnupdateLambda);
	bool bInitZeta = (!cfg_pathModelRead) || (cfg_bInitValue && !cfg_bUnupdateZeta);

	Vec<double> vInitParams(func.GetParamNum());
	/// if the model are inputed, then using the input parameters
	if (cfg_pathModelRead) {
		func.GetParam(vInitParams.GetBuf());
	}
	if (bInitWeight)  {
		lout << "[Init Parameters] Zero" << endl;
		vInitParams.Fill(0);
	}
	if (bInitZeta) {
		for (int i = 0; i <= m.GetMaxLen(); i++) {
			vInitParams[m.GetParamNum() + i] = max(0, i - 1) * log(m.m_pVocab->GetSize()); // set zeta
		}
	}


	solve.Run(vInitParams.GetBuf());

	// Finish
	m.WriteT(cfg_pathModelWrite);

	SAFE_DELETE(pTrain);
	SAFE_DELETE(pValid);
	SAFE_DELETE(pTest);

	SAFE_DELETE(pv);

	return 1;

	
	
}

#endif
