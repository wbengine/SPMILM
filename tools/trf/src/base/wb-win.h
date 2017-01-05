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
* \file
* \author WangBin
* \date 2016-04-28
* \brief Provide the toolkits for cmd window of window platform.
*
* the tools include:
*	+# title toolkits, output info to the title of cmd window (such as precent(%) information)
*/



#pragma once
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>
#ifndef __linux
#include <conio.h>
#endif
using namespace std;
#include "wb-vector.h"




/// Debug
#define F_RETURN(b) {if(!(b)) {cout<<"<F_RETURN>Error = "<<__FILE__<<" (line "<<__LINE__<<")"<<endl; return 0;} }


namespace wb
{
    /** \addtogroup system
    @{
    */

	const int cn_title_max_len = 500;

	/**
	 * \author WangBin
	 * \date 2016-04-28
	 * \brief title class - output to the title
	 */
	class Title
	{
	public:
		static string m_global_title; ///< the global title. All the other information are follow the global title
		static long long m_precent_max; ///< for Precent function, the maxiumn value
		static long long m_precent_cur; ///< for Precent function, the current value
		static int m_precent_last; ///< record the last output precent value (from 0 to 100)
		static string m_precent_label; ///< the precent output label
	public:
		/// set the global title
		static void SetGlobalTitle(const char *pstr);
		/// get the global title
		static const char* GetGlobalTitle();
		/// output string to title
		static void Puts(const char *pStr);
		/// output precent to title
		static void Precent(long long n = m_precent_cur+1, bool bNew = false, long long nTotal = 100, const char* label = "");
		/// output precent when reading files
		static void Precent(ifstream &ifile, bool bNew = false, const char* label = "");
		/// output precent when reading files
		static void Precent(FILE *fp, bool bNew = false, const char* label = "");
		/// output rate to title
		static void Fraction(long long n = m_precent_cur + 1, bool bNew = false, long long nTotal = 100, const char* label = "");
	};

	/**
	 * \author WangBin
	 * \date 2016-04-28
	 * \brief clock - used to record the time
	 */
	class Clock
	{
	protected:
		clock_t m_nBeginTime;
		clock_t m_nEndTime;
		bool m_bWork;
	public:
		Clock(void);
		~Clock(void);
		/// clean the clock
		void Clean();
		/// begin to record
		clock_t Begin();
		/// record end and return the time
		clock_t End();
		/// get the time, but don't stop recording
		clock_t Get();
		/// wait for sveral millissecond
		void Sleep(clock_t n);
		/// transform the clock_t to second
		static double ToSecond(clock_t t)
		{
#ifdef __linux
			return 1.0*t / omp_get_max_threads() / CLOCKS_PER_SEC;
#else
			return 1.0*t / CLOCKS_PER_SEC;
#endif
		}
	};

#ifndef _linux

#define MAX_PATH_LEN 256
	/*!
	 * \author WangBin
	 * \brief Analize the path including "*" and "?" and "+" symbols.
	 * \date 2016-05-05
	 *
	 * The detial description:
	 *  for example: \n
	 *    if the input path is d:\*.txt+d:\?.bat, \n
	 *    frist, we split the string to 2 paths "d:\*.txt" and "d:\?.bat".\n
	 *    Then we find the corresponding files, i.e. all the "txt" file in "d:\" \n
	 *    and all the ".bat" files whose file name is only one character ("a.bat" or "b.bat").
	 */
	class Path
	{
	protected:
		string m_input; ///< stroe the input string
		Queue<char*> *m_paFiles; ///< store all the files find in the queue
	public:
		Path(const char *path=NULL);
		~Path();

		/// reset the path
		void Reset(const char *path = NULL);
		/// Get pathes
		bool GetPath(char *path);
	private:
		/// Search the files
		void SearchFiles();
	};
#endif //__linux

	/// pause
	void Pause();
	/// print precent in the cmd window
	void outPrecent(long long n, bool bNew = false, long long nTotal = 100, const char* title = "Process");
	/// print precent in the cmd window
	void outPrecent(ifstream &ifile, bool bNew = false, const char* title = "Process");
	/// print precent in the cmd window
	void outPrecent(FILE *fp, bool bNew = false, const char* title = "Process");

	/** @} */
}
