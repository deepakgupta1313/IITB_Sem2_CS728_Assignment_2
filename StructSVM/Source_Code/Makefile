# Makefile for svmhmm SVM-struct API, 6 / 12 / 06

include makefile.template

CFLAGS = $(CXXFLAGS)

all: svm_hmm_learn_hideo svm_hmm_classify

.PHONY: clean clean-all help
help:
	echo "make {clean all svm_hmm_learn_{hideo,loqo} svm_hmm_classify}\n";

#just the top-level directory
clean: svm_light_clean svm_struct_clean
	rm -f *.o *.tcov *.d core core.* gmon.out *.stackdump

#-----------------------#
#----   SVM-light   ----#
#-----------------------#

svm_light_hideo_noexe:
	cd svm_light; make svm_learn_hideo_noexe

svm_light_loqo_noexe:
	cd svm_light; make svm_learn_loqo_noexe

svm_light_clean:
	cd svm_light; make clean

#----------------------#
#----  SVM-STRUCT  ----#
#----------------------#

svm_struct_noexe:
	cd svm_struct; make svm_struct_noexe

svm_struct_clean:
	cd svm_struct; make clean

#-----------------#
#---  SVM-HMM  ---#
#-----------------#

# hideo and loqo are interchangeable optimization packages/routines that can be used by svmlight

svm_hmm_classify: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_classify.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o
	$(LD) $(LDFLAGS) svm_struct_api.o svm_struct/svm_struct_classify.o svm_light/svm_common.o svm_struct/svm_struct_common.o -o $@ $(LIBS)

svm_hmm_learn_loqo: svm_light_loqo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o
	$(LD) $(LDFLAGS) svm_struct/svm_struct_learn.o svm_struct_api.o svm_light/svm_loqo.o svm_light/pr_loqo/pr_loqo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o -o $@ $(LIBS)

svm_hmm_learn_hideo: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o
	$(LD) $(LDFLAGS) svm_struct/svm_struct_learn.o svm_struct_api.o svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o -o svm_hmm_learn $(LIBS)


svm_struct_api.o: svm_struct_api.cpp svm_struct_api.h svm_struct_api_types.h svm_struct/svm_struct_common.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

