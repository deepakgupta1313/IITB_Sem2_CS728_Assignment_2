/***********************************************************************/
/*                                                                     */
/*   svm_struct_api_types.h                                            */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 13.10.03                                                    */
/*                                                                     */
/*   Copyright (c) 2003  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#ifndef svm_struct_api_types
#define svm_struct_api_types

#include <vector>
using std::vector;
#include <string>
using std::string;
#include <iostream>
using std::istream;
#include <stdexcept>
using std::invalid_argument;
#define BOOST_ENABLE_ASSERT_HANDLER //call a user-defined handler when an assert inside BOOST gets triggered
#include <boost/shared_ptr.hpp>
using boost::shared_ptr;
extern "C"
{
#include "svm_light/svm_common.h"
#include "svm_light/svm_learn.h"
}

#define INST_NAME          "SVM-HMM"
#define INST_VERSION       "v2.13"
#define INST_VERSION_DATE  "10 / 11 / 06"

/* default precision for solving the optimization problem */
#define DEFAULT_EPS         0.1
/* default loss rescaling method: 1=slack_rescaling, 2=margin_rescaling */
#define DEFAULT_RESCALING   2
/* default loss function: */
#define DEFAULT_LOSS_FCT    1 //Hamming loss; necessary for hmm-svm Viterbi to work
/* default optimization algorithm to use: */
# define DEFAULT_ALG_TYPE    4
/* store Psi(x,y) once instead of recomputing it every time: */
# define USE_FYCACHE         1
/* max number of input examples: a hack */
#define MAX_NUM_EXAMPLES 10000000

typedef string tag; //tag, label, state
typedef unsigned int tagID; //smaller to store than the full string

/*
if t is in the map,
return a newly assigned unique tag ID
*/
extern tagID registerTag(const tag& t);
/*
return the number of tags that have been registered
(registering is done while reading input)
*/
extern unsigned int getNumTags();
extern const tag& getTagByID(tagID id) throw(invalid_argument);

/*
auxiliary to read_struct_examples()
*/
class strMatcher
{
	public:

		string str;

		strMatcher(const string& s) : str(s) {}
};
strMatcher match(const string& s);

/*
auxiliary to read_struct_examples(): try to match a string literal in an input stream

the stream may be partially read if an error occurs
*/
istream& operator >> (istream& in, const strMatcher& m);

/*
a token is an element of the observable HMM output
*/
class token
{
	public:

		token();
		explicit token(const string& s);
		token(const token& t);
		~token();

		const string& getString() const {return str;}
		//the only way to manipulate the feature list
		SVECTOR& getFeatureMap() {return *features;}

		void setString(const string& s) {str = s;}

		/*
		dot product of our (sparse) feature vector with this (non-sparse) weight vector
		*/
		double dotProduct(const double* weights) const {return sprod_ns(const_cast<double*>(weights), features.get());}

		const token& operator = (const token& t);

	private:

		/*
		initialize the features map/list

		should only be called from a constructor
		*/
		void initFeatures();

		string str; //textual representation (can be empty)
		shared_ptr<SVECTOR> features;
};

typedef class pattern {
  /* this defines the x-part of a training example, e.g. the structure
     for storing a natural language sentence in NLP parsing */
	public:

		pattern() : emissions(new vector<token>()) {}
		pattern(const pattern& p) : emissions(p.emissions) {}
		~pattern() {}

  		unsigned int getLength() const {return emissions->size();}
  		const token& getToken(unsigned int index) const {return (*emissions)[index];}
  		token& getToken(unsigned int index) {return (*emissions)[index];}
  		token& getLastToken() {return emissions->back();}

  		void appendToken(const token& t) {emissions->push_back(t);}

  		void setEmissionsVector(shared_ptr<vector<token> > e) {emissions = e;}

  		const pattern& operator = (const pattern& p) {emissions = p.emissions; return *this;}

  	private:

  		shared_ptr<vector<token> > emissions;
} PATTERN;

typedef class label {
  /* this defines the y-part (the label) of a training example,
     e.g. the parse tree of the corresponding sentence. */

	public:

		label() : tags(new vector<tagID>()) {}
		label(const label& l) : tags(l.tags) {}
		~label() {}

		bool isEmpty() const {return tags->empty();} //see empty_label() in pos_tagging_api.cpp
		bool operator == (const label& l) const;

		unsigned int getLength() const {return tags->size();}
		tagID getTag(unsigned int index) const {return (*tags)[index];}
		tagID& getTag(unsigned int index) {return (*tags)[index];}
		tagID& getLastTag() {return tags->back();}

		void appendTag(tagID id) {tags->push_back(id);}
		//be careful calling these!
		void setLength(unsigned int len) {tags->resize(len);}
		void setTag(unsigned int index, const tagID id) {(*tags)[index] = id;}
		void setTagsVector(shared_ptr<vector<tagID> > t) {tags = t;}

		const label& operator = (const label& l) {tags = l.tags; return *this;}

	private:

		shared_ptr<vector<tagID> > tags;
} LABEL;

typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
} STRUCTMODEL;

typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve
				  quadratic program */
  double newconstretrain;      /* number of new constraints to
				  accumulate before recomputing the QP
				  solution */
  int    ccache_size;          /* maximum number of constraints to
				  cache for each example (used in w=4
				  algorithm) */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][300]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm,
				  2 -> L2-norm */
  int    loss_type;            /* selected loss function from -r
				  command line option. Select between
				  slack rescaling (1) and margin
				  rescaling (2) */
  int    loss_function;        /* select between different loss
				  functions via -l command line
				  option */
  /* further parameters that are passed to init_struct_model() */
  unsigned int featureSpaceSize; //number of features for a word
} STRUCT_LEARN_PARM;

typedef struct struct_test_stats {
  /* you can add variables for keeping statistics when evaluating the
     test predictions in svm_struct_classify. This can be used in the
     function eval_prediction and print_struct_testing_stats. */
  unsigned int numTokens, numCorrectTags; //for calculating average loss
} STRUCT_TEST_STATS;

#endif
