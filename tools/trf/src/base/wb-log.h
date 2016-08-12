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
* \brief a definition of a class Log, which can output to the cmd window and the log file simultaneously.
* \detials  
* In wb-log.cpp, there are a Log variable "lout", which can be directly used just like "cout". For example:
*	\code{.cpp}
*		lout<<"Hello Wrold!"<<endl;
*	\endcode
* "lout" has the following advantages:
*		-# It can output to the cmd and the log file. Besides they can be controled.
*		-# It can output the begin and end date of the program.
*		-# If the output varialbes are bools, then it will output true/false
*		-# If the output variables are char*, then it will detect if the pointer is NULL.
*		-# There are some macros, such as lout_variable, lout_error, lout_waring and so on;
*/

#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <cstring>
using namespace std;
#include "wb-linux.h"

/**
 * \brief define all the code written by Bin Wang.
*/
namespace wb
{

#define wbLog_LevelSpace "\t"

#define wbLog_OutputSpace(io) \
	for (int i=0; i<m_nLevel; i++) io<<wbLog_LevelSpace;
#define wbLog_Output(x) \
	if (m_bOutputLog) { {wbLog_OutputSpace(m_fileLog)} m_fileLog<<x; } \
	if (m_bOutputCmd) { {wbLog_OutputSpace(cout)} cout<<x; }\
	return *this;

	/// a progress bar class
	class ProgressBar
	{
	private:
		long long m_num; ///< current number
		long long m_total; ///< the maximum number
		char m_symbol[3];   ///< the progress bar style. such as "[=]"
		int m_lastprec; ///< save the last precent, if current precent == lastprecent, then we don't update
		const int m_barmaxlen = 50; ///< the length of bar 

		string m_strhead; ///< the string shown in before the bar.
	public:
		ProgressBar() {
			m_total = 100;
			memcpy(m_symbol, "[=]", sizeof(3));
			m_lastprec = -1;
			m_strhead = "";
		}
		ProgressBar(long long total, const char *head = "", const char* sym = "[=]"){
			Reset(total, head, sym);
		}
		/// reset the progress bar
		void Reset(long long total = 100, const char *head = "", const char* sym = "[=]");
		/// reset the progress bar
		void Reset(long long n, long long total = 100, const char *head = "", const char* sym = "[=]");
		/// update the progress bar. n should be from 1 to m_total
		void Update(long long n = -1);
	};

	/**
	* \class Log
	* \author WangBin
	* \date 2013-8-29
	* \brief this class can output to the cmd window and log files simultaneously.
	* \detials
	* In wb-log.cpp, there are a Log variable "lout", which can be directly used just like "cout". For example:
	*	\code{.cpp}
	*		lout<<"Hello Wrold!"<<endl;
	*	\endcode
	* "lout" has the following advantages:
	*		-# It can output to the cmd and the log file. Besides they can be controled.
	*		-# It can output the begin and end date of the program.
	*		-# If the output varialbes are bools, then it will output true/false
	*		-# If the output variables are char*, then it will detect if the pointer is NULL.
	*		-# There are some macros, such as lout_variable, lout_error, lout_waring and so on;
	*		-# Include a progress bar function which can output a progress bar on the console window
	*/
	class Log
	{
	protected:
		ofstream m_fileLog; ///< log file stream
		bool m_bOutputLog; ///< if output to the log file
		bool m_bOutputCmd; ///< if output to the cmd window

		short m_nLevel;  ///< output level

		ProgressBar m_bar; ///< the build-in progerss bar.
	public:
		/// constructor
		Log();
		/// destructor
		~Log();
		/// relocate the log file. The defualt log file is "program name".log
		/** \param [in] path the file path
			\param [in] bNew setting true will clean the log file.
		 */
		void ReFile(const char *path, bool bNew = true);
		/// if output to the cmd window
		bool &bOutputCmd();
		/// if output to the log file
		bool &bOutputLog();
		//@{
		/// output.
		Log &operator << (ostream& (*op) (ostream&));
		//template <typename _Arg>
		//Log &operator << (const _Smanip<_Arg>& _Manip) { cout << _Manip; return *this; }
		Log &operator << (int x);
		Log &operator << (short x);
		Log &operator << (long x);
		Log &operator << (long long x);
		Log &operator << (unsigned int x);
		Log &operator << (unsigned short x);
		Log &operator << (unsigned long x);
		Log &operator << (float x);
		Log &operator << (double x);
		Log &operator << (char x);
		Log &operator << (const char* x);
		Log &operator << (const void* x);
		Log &operator << (bool x);
		Log &operator << (string &x);
		//@}
		/// output an array
		template <typename T>
		Log &output(T *pArray, int n, const char *pgap=" ");
		/// level down
		void LevelDown();
		/// level up
		void LevelUp();
		/// progress bar
		void Progress(long long n = -1, bool bInit = false, long long total = 100, const char* head = "");
		/// progress bar for file
		void Progress(FILE *fp, bool bInit = false, const char* head="");
	};

	extern Log lout; ///< the defination is in wb-log.cpp
	
	template <typename T>
	Log &Log::output(T *pArray, int n, const char *pgap/*=" "*/)
	{
		for (int i = 0; i < n; i++) {
			*this << pArray[i] << pgap;
		}
		return *this;
	}


#define lout_variable(x) {wb::lout<<#x" = "<<x<<endl;}
#define lout_variable_precent(x,y) {wb::lout<<#x" = "<<x<<" ("<<100.0*(x)/(y)<<"%)  /"#y<<"="<<y<<endl;}
#define lout_variable_rate(x,y) {wb::lout<<100.0*(x)/(y)<<"%("<<x<<"/"<<y<<")";}
#define lout_array(x, n) {wb::lout<<#x"=[ "; for(int i=0; i<n; i++) wb::lout<<x[i]<<" "; wb::lout<<"]"<<endl; }
#define lout_error(x) {wb::lout.bOutputCmd()=true; wb::lout<<"[ERROR] "<<x<<endl; exit(0);}
#define lout_warning(x) {wb::lout<<"[WARNING] "<<x<<endl;}
#define lout_assert(p) {if(!(p)) lout_error("! ("#p")"); }

#define precent(x, n) (x)<<"("<<100.0*(x)/(n)<<"%)"

	

	


}


