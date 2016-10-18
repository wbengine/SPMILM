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

//#include "hrf-sa-train.h"
#include "hrf-sams.h"
using namespace hrf;

char *cfg_pathVocab = NULL;
int cfg_nHLayer = 1;
int cfg_nHNode = 2;
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
char *cfg_gamma_lambda = "0,0.8";
char *cfg_gamma_hidden = "100,0.8";
char *cfg_gamma_zeta = "0,0.6";
char *cfg_gamma_var = "0,0.8";
float cfg_fMomentum = 0;
float cfg_var_gap = 1e-4;
float cfg_dir_gap = 1;
float cfg_zeta_gap = 10;
bool cfg_bUpdateLambda = false;
bool cfg_bUpdateZeta = false;
int cfg_nAvgBeg = 0;

float cfg_fRegL2 = 0;

bool cfg_bInitValue = false;
bool cfg_bZeroInit = false;
int cfg_nPrintPerIter = 100;
bool cfg_bUnprintTrain = false;
bool cfg_bUnprintValid = false;
bool cfg_bUnprintTest = false;
char *cfg_strWriteAtIter = NULL;

char *cfg_pathWriteMean = NULL;
char *cfg_pathWriteVar = NULL;

Option opt;

_wbMain
{
	opt.Add(wbOPT_STRING, "vocab", &cfg_pathVocab, "The vocabulary");
	opt.Add(wbOPT_STRING, "feat", &cfg_pathFeatStyle, "a feature style file. Set this value will disable -order");
	opt.Add(wbOPT_INT, "order", &cfg_nFeatOrder, "the ngram feature order");
	opt.Add(wbOPT_INT, "len", &cfg_nMaxLen, "the maximum length of HRF");
	opt.Add(wbOPT_INT, "layer", &cfg_nHLayer, "the hidden layer of HRF");
	opt.Add(wbOPT_INT, "node", &cfg_nHNode, "the hidden node of each hidden layer of HRF");
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
	opt.Add(wbOPT_STRING, "gamma-hidden", &cfg_gamma_hidden, "learning rate of VHmatrix");
	opt.Add(wbOPT_STRING, "gamma-zeta", &cfg_gamma_zeta, "learning rate of zeta");
	opt.Add(wbOPT_STRING, "gamma-var", &cfg_gamma_var, "learning rate of variance");
	opt.Add(wbOPT_FLOAT, "momentum", &cfg_fMomentum, "the momentum");
	opt.Add(wbOPT_TRUE, "update-lambda", &cfg_bUpdateLambda, "update lambda");
	opt.Add(wbOPT_TRUE, "update-zeta", &cfg_bUpdateZeta, "update zeta");
	opt.Add(wbOPT_INT, "tavg", &cfg_nAvgBeg, ">0 then apply averaging");
	opt.Add(wbOPT_FLOAT, "vgap", &cfg_var_gap, "the threshold of variance");
	opt.Add(wbOPT_FLOAT, "dgap", &cfg_dir_gap, "the threshold for parameter update");
	opt.Add(wbOPT_FLOAT, "zgap", &cfg_zeta_gap, "the threshold for zeta update");
	opt.Add(wbOPT_FLOAT, "L2", &cfg_fRegL2, "regularization L2");

	opt.Add(wbOPT_TRUE, "init", &cfg_bInitValue, "Re-init the parameters");
	opt.Add(wbOPT_TRUE, "zero-init", &cfg_bZeroInit, "Set the init parameters Zero. Otherwise random init the parameters");
	opt.Add(wbOPT_INT, "print-per-iter", &cfg_nPrintPerIter, "print the LL per iterations");
	opt.Add(wbOPT_TRUE, "not-print-train", &cfg_bUnprintTrain, "donot print LL on training set");
	opt.Add(wbOPT_TRUE, "not-print-valid", &cfg_bUnprintValid, "donot print LL on valid set");
	opt.Add(wbOPT_TRUE, "not-print-test", &cfg_bUnprintTest, "donot print LL on test set");
	opt.Add(wbOPT_STRING, "write-at-iter", &cfg_strWriteAtIter, "write the LL per iteration, such as [1:100:1000]");

	opt.Add(wbOPT_STRING, "write-mean", &cfg_pathWriteMean, "write the expecataion on training set");
	opt.Add(wbOPT_STRING, "write-var", &cfg_pathWriteVar, "write the variance on training set");

	opt.Parse(_argc, _argv);

	lout << "*********************************************" << endl;
	lout << "         TRF_SAtrain.exe { by Bin Wang }        " << endl;
	lout << "\t" << __DATE__ << "\t" << __TIME__ << "\t" << endl;
	lout << "**********************************************" << endl;

	omp_set_num_threads(cfg_nThread);
	lout << "[OMP] omp_thread = " << omp_get_max_threads() << endl;
	trf::omp_rand(cfg_nThread);

	/* Load Model and Vocab */
	Vocab *pv = new Vocab(cfg_pathVocab);
	Model m(pv, cfg_nHLayer, cfg_nHNode, cfg_nMaxLen);
	if (cfg_pathModelRead) {
		m.ReadT(cfg_pathModelRead);
	}
	else {
		m.LoadFromCorpus(cfg_pathTrain, cfg_pathFeatStyle, cfg_nFeatOrder);
	}
	lout_variable(m.m_hlayer);
	lout_variable(m.m_hnode);
	lout_variable(m.GetParamNum());

	/* Load corpus */
	trf::CorpusTxt *pTrain = (cfg_pathTrain) ? new trf::CorpusTxt(cfg_pathTrain) : NULL;
	trf::CorpusTxt *pValid = (cfg_pathValid) ? new trf::CorpusTxt(cfg_pathValid) : NULL;
	trf::CorpusTxt *pTest = (cfg_pathTest) ? new trf::CorpusTxt(cfg_pathTest) : NULL;


	Train *pFunc;
	if (cfg_bUpdateZeta) {
		pFunc = new SAMSZeta;
	}
	else if (cfg_bUpdateLambda) {
		pFunc = new SALambda;
	}

	pFunc->OpenTempFile(cfg_pathModelWrite);
	pFunc->Reset(&m, pTrain, pValid, pTest);
	pFunc->m_nMinibatch = cfg_nMiniBatch;
	pFunc->m_nAvgBeg = cfg_nAvgBeg;
	pFunc->m_fRegL2 = cfg_fRegL2;
	pFunc->m_aPrint[0] = !cfg_bUnprintTrain;
	pFunc->m_aPrint[1] = !cfg_bUnprintValid;
	pFunc->m_aPrint[2] = !cfg_bUnprintTest;
	pFunc->m_nPrintPerIter = cfg_nPrintPerIter;
	VecUnfold(cfg_strWriteAtIter, pFunc->m_aWriteAtIter);
	pFunc->m_nIterMax = cfg_nIterTotalNum; // fix the iteration number

	if (cfg_bUpdateZeta) {
		SAMSZeta *p = (SAMSZeta*)pFunc;
		p->m_zeta_rate.Reset(cfg_gamma_zeta, cfg_t0);
		p->m_zeta_gap = cfg_zeta_gap;
	}
	else if (cfg_bUpdateLambda) {
		SALambda *p = (SALambda*)pFunc;
		p->m_feat_rate.Reset(cfg_gamma_lambda, cfg_t0);
		p->m_hidden_rate.Reset(cfg_gamma_hidden, cfg_t0);
		p->m_dir_gap = cfg_dir_gap;
#ifdef _Var
		p->m_var_rate.Reset(cfg_gamma_var, cfg_t0);
		p->m_var_gap = cfg_var_gap;
#endif
	}

	/* set initial values */
	bool bInit = (!cfg_pathModelRead) || cfg_bInitValue;
	pFunc->Run(bInit);

	// Finish
	m.WriteT(cfg_pathModelWrite);

	SAFE_DELETE(pTrain);
	SAFE_DELETE(pValid);
	SAFE_DELETE(pTest);

	SAFE_DELETE(pv);

	return 1;

}


#endif