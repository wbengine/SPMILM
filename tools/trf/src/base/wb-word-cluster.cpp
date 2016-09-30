#include "wb-word-cluster.h"

namespace wb
{
	void WordCluster::InitCount(const char *path, const char *pTagVocab)
	{
		bool bFound;
		File file(path, "rt");
		char *pLine = NULL;
		m_nSentNum = 0;
		m_nVocabSize = 0;

		m_nUnigramNum = 0;
		m_nBigramNum = 0;

		while (pLine = file.GetLine(true)) {

			Array<int> aWords;

			char *pWord = strtok(pLine, " \t");
			while (pWord) {
				aWords.Add(atoi(pWord));
				pWord = strtok(NULL, " \t");
			}

			m_nSentNum += 1;
			//Unigram
			for (int i = 0; i < aWords.GetNum(); i++) {
				m_nVocabSize = max(m_nVocabSize, aWords[i] + 1);

				int *pCount = m_wordCount.Insert(aWords[i], bFound);
				if (!bFound) { *pCount = 1; m_nUnigramNum++; }
				else (*pCount)++;
			}
			//Bigram
			for (int i = 0; i < aWords.GetNum() - 1; i++) {
				int *pCount = m_wordGramCount.Insert(aWords.GetBuffer(i), 2, bFound);
				if (!bFound) { *pCount = 1; m_nBigramNum++; }
				else (*pCount)++;

				int idx[3];
				idx[0] = aWords[i + 1];
				idx[1] = aWords[i];
				pCount = m_invWordGram.Insert(idx, 2, bFound);
				if (!bFound) *pCount = 1;
				else (*pCount)++;
			}
		}


		lout_assert(m_wordCount.GetNum() >= m_nClassNum);
		lout_variable(m_nVocabSize);
		lout_variable(m_nClassNum);
		lout_variable(m_nUnigramNum);
		lout_variable(m_nBigramNum);

		m_aClass.SetNum(m_nVocabSize);
		m_aClass.Fill(m_nClassNum - 1); ///< 由于存在没有count的word，因此需要为没有cout的词分配一个class

		if (pTagVocab) {
			lout << "Load Tagvocab: " << pTagVocab << endl;
			Read_TagVocab(pTagVocab);
			return;
		}
		//对Unigram排序，初始分类
		Heap<int, int> heap(HEAPMODE_MAXHEAP);
		int w, n;
		int *pCount;
		LHashIter<int, int> iter(&m_wordCount);
		while (pCount = iter.Next(w)) {
			heap.Insert(w, *pCount);
		}
		int nClass = 0;
		while (heap.OutTop(w, n)) {  //前m_nClass-2个word，每个word一个类；后边的所有出现的class赋予m_nClassNum-2类；未出现的赋予m_nClass-1类
			if (nClass <= m_nClassNum - 2)
				m_aClass[w] = nClass;
			else
				m_aClass[w] = m_nClassNum - 2;
			//m_aClass[w] = (nClass >= m_nClassNum-2)? m_nClassNum-2: nClass;
			nClass++;
		}
		//WriteCount(m_wordGramCount, wbFile("test.count", "wt"));
		//计算类别相关的count
		UpdataCount();
	}

	void WordCluster::UpdataCount()
	{
		//wbFile test("a.dbg", "wt");
		//清空count数据
		m_classCount.Fill(0);
		//m_classGramCount.Fill(0);
		for (int i = 0; i < m_nClassNum + 1; i++)
			memset(m_pClassGramCount[i], 0, sizeof(int)*(m_nClassNum + 1));
		m_classWordCount.Fill(0);
		m_wordClassCount.Fill(0);

		int w, n;
		int *pCount;
		//计算类别相关的count
		//Unigram class
		lout << "Update class Unigram" << endl;
		LHashIter<int, int> iter(&m_wordCount);
		int nTempTimes = 0;
		lout.Progress(0, true, m_nUnigramNum, "Update Unigram:");
		while (pCount = iter.Next(w)) {
			CountAdd(m_classCount, m_aClass[w], *pCount);
			lout.Progress(++nTempTimes);
		}
		//Bigram class
		lout << "Update class Bigram" << endl;
		int gram[10];
		int g[10];
		Trie<int, int> *pSub;
		TrieIter2<int, int> iter2(&m_wordGramCount, gram, 2);
		nTempTimes = 0;
		lout.Progress(0, true, m_nBigramNum, "Update Bigram:");
		while (pSub = iter2.Next()) {
			if (!pSub->IsDataLegal())
				continue;

			//test.Print("%d %d\t%d\n",  gram[0], gram[1], *(pSub->GetData()));

			g[0] = m_aClass[gram[0]];
			g[1] = m_aClass[gram[1]];
			CountAdd(m_pClassGramCount, g, 2, *(pSub->GetData()));
			g[0] = m_aClass[gram[0]];
			g[1] = gram[1];
			Reverse(g); //将word储存在前，为了方便遍历某个word的有效前级class
			CountAdd(m_classWordCount, g, 2, *(pSub->GetData()));
			g[0] = gram[0];
			g[1] = m_aClass[gram[1]];
			CountAdd(m_wordClassCount, g, 2, *(pSub->GetData()));
			lout.Progress(++nTempTimes);
		}


		//Sum { N(w)logN(w) }
		lout << "Prepare Sum" << endl;
		m_dWordLogSum = 0;
		LHashIter<int, int> iterW(&m_wordCount);
		while (pCount = iterW.Next(w)) {
			lout_assert(*pCount >= 0);
			if (*pCount != 0)
				m_dWordLogSum += 1.0 * (*pCount) / m_nSentNum * log((double)(*pCount));
		}
	}

	void WordCluster::WriteCount(LHash<int, int> &count, File &file)
	{
		LHashIter<int, int> iter(&count);
		int w;
		int *pCount;
		while (pCount = iter.Next(w)) {
			file.Print("%d\t%d\n", w, *pCount);
		}
	}
	void WordCluster::WriteCount(Trie<int, int> &count, File &file, bool bReverse /*=false*/)
	{
		int gram[10];
		Trie<int, int> *pSub;
		TrieIter2<int, int> iter(&count, gram, 2);
		while (pSub = iter.Next()) {
			if (pSub->IsDataLegal()) {
				if (bReverse)
					file.Print("%d %d\t%d\n", gram[1], gram[0], *(pSub->GetData()));
				else
					file.Print("%d %d\t%d\n", gram[0], gram[1], *(pSub->GetData()));
			}
		}
	}
	void WordCluster::WriteRes_WordClass(const char *path)
	{
		File file(path, "wt");
		for (int i = 0; i < m_aClass.GetNum(); i++) {
			file.Print("%d\t%d\n", i, m_aClass[i]);
		}
	}
	void WordCluster::WriteRes_ClassWord(const char *path)
	{

		Array<Array<int>*> aClass(m_nClassNum);
		for (int i = 0; i < m_nClassNum + 1; i++)
			aClass[i] = new Array<int>();

		int w;
		for (w = 0; w < m_nVocabSize; w++)
		{
			aClass[m_aClass[w]]->Add(w);
		}

		File file(path, "wt");
		for (int i = 0; i < m_nClassNum; i++) {
			file.Print("[%d]\t", i);
			for (int n = 0; n < aClass[i]->GetNum(); n++) {

				int w = aClass[i]->Get(n);
				int *pcount = m_wordCount.Find(w);
				int ncount = (pcount == NULL) ? 0 : *pcount;

				file.Print("%d{%d} ", aClass[i]->Get(n), ncount); //打印出现count
			}
			file.Print("\n");
		}


		for (int i = 0; i < m_nClassNum; i++)
			SAFE_DELETE(aClass[i]);
	}
	void WordCluster::WriteRes_TagVocab(const char *path)
	{
		File file(path, "wt");
		for (int i = 0; i < m_aClass.GetNum(); i++) {
			file.Print("%d\t%d %d\n", i, i, m_aClass[i]);
		}
	}
	void WordCluster::Read_TagVocab(const char *path)
	{
		File file(path, "rt");
		for (int i = 0; i < m_aClass.GetNum(); i++) {
			int g, w, c;
			fscanf(file, "%d\t%d %d\n", &g, &w, &c);
			if (g != i) {
				lout_error("read TagVocab ERROR!!");
			}
			m_aClass[g] = c;
		}
		UpdataCount();
	}

	double WordCluster::LogLikelihood()
	{
		double dSumClassGram = 0;
		double dSumClass = 0;
		double dSumWord = 0;
		// Sum{ N(g_v,g_w)logN(g_v,g_w) }
		// 	int g[10];
		// 	wbTrie<int,int> *pSub;
		// 	wbTrieIter2<int,int> iter2(m_classGramCount, g, 2);
		// 	while (pSub = iter2.Next()) {
		// 		if (!pSub->IsDataLegal())
		// 			continue;
		// 
		// 		int n = *(pSub->GetData());
		// 		lout_assert( n>= 0 );
		// 		if ( n != 0 ) {
		// 			dSumClassGram += 1.0 * n/m_nSentNum * log((double)n);
		// 		}
		// 	}
		for (int i = 0; i < m_nClassNum + 1; i++) {
			for (int j = 0; j < m_nClassNum + 1; j++) {
				int n = m_pClassGramCount[i][j];
				if (n < 0) {
					lout_error("classGramCount (" << n << ") < 0")
				}
				if (n != 0) {
					dSumClassGram += 1.0 * n / m_nSentNum * log((double)n);
				}
			}
		}



		//Sum { N(g)logN(g) }
		int c, w;
		int *pCount;
		LHashIter<int, int> iterC(&m_classCount);
		while (pCount = iterC.Next(c)) {
			lout_assert(*pCount >= 0);
			if (*pCount != 0)
				dSumClass += 1.0 * (*pCount) / m_nSentNum * log((double)(*pCount));
		}

		//Sum { N(w)logN(w) } 
		// 	wbLHashIter<int,int> iterW(&m_wordCount);
		// 	while (pCount = iterW.Next(w)) {
		// 		lout_assert(*pCount>=0);
		// 		if ( *pCount != 0 )
		// 			dSumWord += 1.0 * (*pCount) / m_nSentNum * log((double)(*pCount));
		// 	}

		double dRes = dSumClassGram - 2 * dSumClass + m_dWordLogSum;
		return dRes;
	}

	void WordCluster::MoveWord(int nWord, bool bOut /* = true */)
	{
		int nClass = m_aClass[nWord];
		int sig = (bOut) ? -1 : 1;

		// class unigram
		int *pCount = m_wordCount.Find(nWord);
		if (pCount == NULL)
			return;
		CountAdd(m_classCount, nClass, sig*(*pCount));


		// class bigram
		int g[10];
		int w[10];

		g[1] = nClass;
		Trie<int, int> *pSub = m_classWordCount.FindTrie(&nWord, 1);
		if (pSub) {  //遍历所有可能的前继class
			TrieIter<int, int> iter(pSub);
			Trie<int, int> *p;
			while (p = iter.Next(g[0])) {
				CountAdd(m_pClassGramCount, g, 2, sig*(*p->GetData()));
			}
		}

		g[0] = nClass;
		pSub = m_wordClassCount.FindTrie(&nWord, 1);
		if (pSub) { //遍历所有可能的后继class
			TrieIter<int, int> iter(pSub);
			Trie<int, int> *p;
			while (p = iter.Next(g[1])) {
				CountAdd(m_pClassGramCount, g, 2, sig*(*p->GetData()));
			}
		}

		g[0] = nClass;
		g[1] = nClass;
		w[0] = nWord;
		w[1] = nWord;
		pCount = m_wordGramCount.Find(w, 2);
		if (pCount) {
			CountAdd(m_pClassGramCount, g, 2, *pCount); //加上count
		}

		// word class pair

		int v;
		g[1] = nClass;

		// 	int a1=0, a2=0;
		// 	int b1=0, b2=0;
		// 	wbLHashIter<int,int> vocabIter(&m_wordCount);
		// 	while (vocabIter.Next(v)) { //遍历所有的词v
		// 		g[0] = v;
		// 
		// 		w[0] = v;
		// 		w[1] = nWord;
		// 		pCount = m_wordGramCount.Find(w, 2);
		// 		if (pCount) {
		// 			a1 ++;
		// 			CountAdd(m_wordClassCount, g, 2, sig*(*pCount));
		// 		}
		// 
		// 		w[0] = nWord;
		// 		w[1] = v;
		// 		pCount = m_wordGramCount.Find(w, 2);
		// 		if (pCount) {
		// 			b1 ++;
		// 			CountAdd(m_classWordCount, g, 2, sig*(*pCount));
		// 		}
		// 	}

		//遍历nWord的前继
		pSub = m_invWordGram.FindTrie(&nWord, 1);
		if (pSub) {
			TrieIter<int, int> iter(pSub);
			Trie<int, int> *p;
			while (p = iter.Next(v)) {
				g[0] = v;
				w[0] = v;
				w[1] = nWord;
				//			pCount = m_wordGramCount.Find(w, 2);
				// 			if ( *pCount != *(p->GetData()) ) {
				// 				lout_error("ERROR_test");
				// 			}
				pCount = p->GetData();
				if (pCount) {
					//a2++;
					CountAdd(m_wordClassCount, g, 2, sig*(*pCount));
				}
			}
		}
		//遍历nWord后继
		pSub = m_wordGramCount.FindTrie(&nWord, 1);
		if (pSub) {
			TrieIter<int, int> iter(pSub);
			Trie<int, int> *p;
			while (p = iter.Next(v)) {
				g[0] = v;
				pCount = p->GetData();
				if (pCount) {
					//b2++;
					CountAdd(m_classWordCount, g, 2, sig*(*pCount));
				}
			}
		}

		// 	lout_variable(a1);
		// 	lout_variable(a2);
		// 	lout_variable(b1);
		// 	lout_variable(b2);
		// 	Pause();
	}

	void WordCluster::ExchangeWord(int nWord, int nToClass)
	{
		if (nToClass == m_aClass[nWord])
			return;

		MoveWord(nWord, true); // move out from nClass
		m_aClass[nWord] = nToClass;
		MoveWord(nWord, false); // move into nToClass

		// 	m_aClass[nWord] = nToClass;
		// 	UpdataCount();
	}

	void WordCluster::Cluster(int nMaxTime /* = -1 */)
	{
		bool bChange = false;
		int nTimes = 0;


		lout_variable(m_nVocabSize);
		int nTotalSwitchClassNum = m_nClassNum;
		//将未出现的word分到同一类
		for (int w = 0; w < m_nVocabSize; w++) {
			if (m_wordCount.Find(w) == NULL) {
				m_aClass[w] = m_nClassNum - 1; ///< 赋予最后一个类
				nTotalSwitchClassNum = m_nClassNum - 1;
			}
		}

		while (1)  //外层循环
		{
			bChange = false;
			int w;
			// 		wbLHashIter<int,int> vocabIter(&m_wordCount);
			// 		while (vocabIter.Next(w)) //遍历每个word
			for (w = 0; w < m_nVocabSize; w++)
			{
				if (m_wordCount.Find(w) == NULL)
					continue;

				int nOldClass = m_aClass[w]; //保存目前的class
				int nOptClass = -1;
				double dOptValue = -1e22; //对数似然值

				for (int c = 0; c < nTotalSwitchClassNum; c++) //转移到每个class
				{
					ExchangeWord(w, c);
					double dCurValue = LogLikelihood(); //替换后的负对数似然值
					if (dCurValue > dOptValue) {
						dOptValue = dCurValue;
						nOptClass = c;
					}
				}
#ifdef _DEBUG
				//lout<<"class_"<<nOldClass<<" -> class_"<<nOptClass<<endl;
#endif
				ExchangeWord(w, nOptClass);
				if (nOptClass != nOldClass) {//改变了类
					lout << "[exchange_" << nTimes + 1 << "] " << w << " class_" << nOldClass << " -> class_" << nOptClass << " value=" << dOptValue << endl;
					bChange = true;
					if (nOptClass >= nTotalSwitchClassNum) {
						lout_error("[cluster] 未定义的to class-id (" << nOptClass << ") for word (" << w << ") ");
					}
					/*Pause();*/
				}
			}

			lout_variable(LogLikelihood());
			//统计下每个class中的词数目
			Array<int> aNum(m_nClassNum);
			aNum.Fill(0);
			// 		vocabIter.Init();
			// 		while (vocabIter.Next(w)) {
			for (w = 0; w < m_nVocabSize; w++) {
				if (m_aClass[w] >= m_nClassNum) {
					lout_error("[cluster] 未定义的class-id (" << m_aClass[w] << ") for word (" << w << ") ");
				}
				aNum[m_aClass[w]] ++;
			}

			for (int i = 0; i < m_nClassNum; i++)
				lout << i << "[" << aNum[i] << "] ";
			lout << endl;

			//打印当前的结果
			if (m_pathWordClass)
				WriteRes_WordClass(m_pathWordClass);
			if (m_pathClassWord)
				WriteRes_ClassWord(m_pathClassWord);
			if (m_pathTagVocab)
				WriteRes_TagVocab(m_pathTagVocab);

			nTimes++;
			if (nTimes == nMaxTime) {
				lout << "[end] Get Max Times" << endl;
				break;
			}
			if (!bChange) {
				lout << "[end] No Change" << endl;
				break;
			}
		}


	}

	void WordCluster::SimpleCluster()
	{
		// 排序
		Heap<int, int> heap(HEAPMODE_MAXHEAP);
		for (int w = 0; w < m_nVocabSize; w++) {
			int *p = m_wordCount.Find(w);
			int nTemp = 0;;
			if (p == NULL)
				nTemp = 0;
			else
				nTemp = (int)sqrt((double)(*p));  ///< 对词频计算平方根
			heap.Insert(w, nTemp);
		}


		int n = -1;
		int w, count, preCount = -1;
		while (heap.OutTop(w, count)) {

			//确定当前的class
			if (count != preCount) {
				preCount = count;
				n++;
			}


			if (n >= m_nClassNum)
				m_aClass[w] = m_nClassNum - 1;
			else
				m_aClass[w] = n;

		}
	}
}