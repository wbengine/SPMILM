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
* \date 2016-05-05
* \brief Define the option class
*/

#pragma once
#include "wb-log.h"
#include "wb-vector.h"
#include "wb-win.h"

namespace wb
{
	/// define the value type
	typedef enum {
		wbOPT_TRUE,			///< is true if exist
		wbOPT_FALSE,			///< set false if exist
		wbOPT_INT,			///< integer
		wbOPT_STRING,			///< string
		wbOPT_FLOAT			/// float
	} ValueType;

	/// structure of the value
	typedef struct {
		ValueType type;			///< value type
		const char* pLabel;		///< label content. Donot stroe the flag "-"
		void* pAddress;			///< value memory address
		const char* pDocMsg;	///< value usage docment
	} Opt_Struct;


	/**
	 * \author WangBin
	 * \date 2016-05-05
	 * \brief Get the option from command line or command files
	 */
	class Option
	{
	public:
		Array<Opt_Struct> m_opts; ///< all the options
		string m_strOtherHelp; ///< extra help information, which will be output in PrintUsage
		bool m_bOutputValues; ///< if output value after get options from the command line or file
		bool m_bMustCommand; ///< setting 'true' means that, report error when no option input.

		Array<char*> m_allocedBufs; ///< if read from file, we may need to allocate memory for string.
	public:
		/// constructor
		Option();
		/// destructor
		/* If the option type is string, then 'strdup' is used to create a new string.
		   As a result, in destructor, we should 'free' these string.
		*/
		~Option();
		/// Add a option
		void Add(ValueType t, const char* pLabel, void *pAddress, const char* pDocMsg = NULL);
		/// output usage
		void PrintUsage();
		/// output a value
		void Print(ValueType type, void* pAddress);
		/// output values
		void PrintValue();
		/// parse a single option, "pvalue" can be NULL
		void Parse(const char *plabel, const char *pvalue);
		/// get the options from command line
		/**
		* \param [in] argc command line count
		* \param [in] argv command line strings 
		* \param [in] pOtherHelp if parse fails, output the help information
		* \param [in] bOutValue if set true, then output the values
		* \param [in] bMustCommand if set true, then report error when there are on command inputs
		* \return the option number
		*/
		int Parse(int argc, char **argv);

		///< get the options from a file. the file format is: [label] = [values] in each line.
		/**
		* \param [in] optfile the option file name
		* \param [in] pOtherHelp if parse fails, output the help information
		* \param [in] bOutValue if set true, then output the values
		* \param [in] bMustCommand if set true, then report error when there are on command inputs
		* \return the option number
		*/
		int Parse(const char *optfile);
	};
}
