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

/**
* \dir
* \author wangbin
* \date 2014-03-24
* \todo word cluster
* \brief word cluster
*/
/**
* \file
* \author wangbin
* \date 2014-03-24
* \brief [main]
*/

#include "wb-word-cluster.h"
using namespace wb;

static char *cfg_pathTxt = NULL;
static char *cfg_pathWordClass = NULL;
static char *cfg_pathClassWord = NULL;
static char *cfg_pathTagVocab = NULL;
static int cfg_nClassNum = 10;
static bool cfg_bSimpleCluster = false;
static char *cfg_pathReadTagVocab = NULL;

Option opt;

_wbMain
{
	opt.Add(wbOPT_STRING, "txt", &cfg_pathTxt, "input txt(word id, begin from 0)");
	opt.Add(wbOPT_STRING, "out-wc", &cfg_pathWordClass, "output the cluster file, [word_id, class_id]");
	opt.Add(wbOPT_STRING, "out-cw", &cfg_pathClassWord, "output the cluster file, [class_id, word_id1, word_id2,...]");
	opt.Add(wbOPT_STRING, "tag-vocab", &cfg_pathTagVocab, "output a tag-vocab file? [word_id, word_id, class_id]");
	opt.Add(wbOPT_INT, "num", &cfg_nClassNum, "class num");
	opt.Add(wbOPT_TRUE, "simple-cluster", &cfg_bSimpleCluster, "just using the count of unigram to perform cluster");
	opt.Add(wbOPT_STRING, "read-tag-vocab", &cfg_pathReadTagVocab, "read tag-vocab, calculate the likelihood");

	opt.Parse(_argc, _argv);

	WordCluster cluster(cfg_nClassNum);
	cluster.m_pathWordClass = cfg_pathWordClass;
	cluster.m_pathClassWord = cfg_pathClassWord;
	cluster.m_pathTagVocab = cfg_pathTagVocab;

	cluster.InitCount(cfg_pathTxt, cfg_pathReadTagVocab);
	// 	if ( cfg_pathReadTagVocab ) {
	// 		cluster.Read_TagVocab(cfg_pathReadTagVocab);
	// 		lout_variable(cluster.LogLikelihood());
	// 		return 1;
	// 	}


	if (cfg_bSimpleCluster) {
		lout << "Simple Cluster..." << endl;
		cluster.SimpleCluster();
	}
	else {
		lout << "Cluster..." << endl;
		cluster.Cluster(100);
	}

	if (cfg_pathWordClass)
		cluster.WriteRes_WordClass(cfg_pathWordClass);
	if (cfg_pathClassWord)
		cluster.WriteRes_ClassWord(cfg_pathClassWord);
	if (cfg_pathTagVocab)
		cluster.WriteRes_TagVocab(cfg_pathTagVocab);

	// 	cluster.m_aClass[0] = 9;
	// 	cluster.m_aClass[10] = 0;
	// 	cluster.UpdataCount();
	// 	cluster.WriteClass(cfg_pathClass);
	// 	cluster.WriteCount(cluster.m_wordCount, wbFile("unigram.count", "wt"));
	// 	cluster.WriteCount(cluster.m_wordGramCount, wbFile("bigram.count", "wt"));
	// 	cluster.WriteCount(cluster.m_classCount, wbFile("classUnigram.count", "wt") );
	// 	cluster.WriteCount(cluster.m_classGramCount, wbFile("classBigram.count", "wt") );
	// 	cluster.WriteCount(cluster.m_classWordCount, wbFile("classWord.count", "wt"), true );
	// 	cluster.WriteCount(cluster.m_wordClassCount, wbFile("wordClass.count", "wt") );

	// 	cluster.ExchangeWord(0, 9);
	// 	cluster.ExchangeWord(10,0);
	// 	cluster.WriteClass("class1.txt");
	// 	cluster.WriteCount(cluster.m_wordCount, wbFile("unigram1.count", "wt"));
	// 	cluster.WriteCount(cluster.m_wordGramCount, wbFile("bigram1.count", "wt"));
	// 	cluster.WriteCount(cluster.m_classCount, wbFile("classUnigram1.count", "wt") );
	// 	cluster.WriteCount(cluster.m_classGramCount, wbFile("classBigram1.count", "wt") );
	// 	cluster.WriteCount(cluster.m_classWordCount, wbFile("classWord1.count", "wt"), true);
	// 	cluster.WriteCount(cluster.m_wordClassCount, wbFile("wordClass1.count", "wt") );


	return 1;
};
