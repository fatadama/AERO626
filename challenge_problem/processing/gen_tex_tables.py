"""@package gen_tex_tables
Generate tables in LaTeX format for insertion into writeup
"""
import os
import numpy as np

class filterdat():
	def __init__(self,name,notes = None):
		## identifying name
		self.name = name
		## 3 x 4 array of data; each coumn is MSE1, MSE2, percentage conservative, percent overconfident
		self.data = np.zeros((4,3,4))
		## notes: if nonempty, a string that's printed nextt to the fkilter name in the table's label
		self.notes = notes
	def set_data(self,Ts,namebit,FID):
		line = FID.readline()
		dataList = line.split(',')
		# remove endline
		dataList[3] = dataList[3][0:len(dataList[3])-1]
		if Ts == 0.01:
			uind = 0
		elif Ts == 0.1:
			uind = 1
		elif Ts == 1.0:
			uind = 2
		nind = namebit
		self.data[nind,uind,0] = float(dataList[0])
		self.data[nind,uind,1] = float(dataList[1])
		self.data[nind,uind,2] = float(dataList[2])
		self.data[nind,uind,3] = float(dataList[3])
	def print_out(self,FID=None):
		TS = [0.01,0.1,1.0]
		if FID is None:
			for nb in range(4):
				# print to terminal
				print('\begin{\table}[h!]')
				print('\begin{tabular}{|c|c|c|c|c|}')
				for k in range(3):
					print('\hline')
					print('%8.4g & %8.4g & %8.4g & %8.4g & %8.4g \\' % (TS[k],self.data[nb,k,0],self.data[nb,k,1],self.data[nb,k,2],self.data[nb,k,3]))
				print('\end{tabular}')
				print('\caption{Performance metrics for ')
				print(' %s ' % (self.name.upper()))
				if self.notes is not None:
					print('(%s) ' % self.notes)
				if nb == 0:
					print('with no forcing}')
				if nb == 1:
					print('with Gaussian forcing term}')
				if nb == 2:
					print('with cosine forcing term}')
				if nb == 3:
					print('with Gaussian and cosine forcing terms}')
				print('\end{table}')
		else:
			for nb in range(4):
				FID.write('\n\n%%************************************************\n%% Performance table for filter %s with namebit %d\n' % (self.name,nb))
				# print to terminal
				FID.write('\\begin{table}[h!]\n')
				FID.write('\\centering\n')
				FID.write('\\begin{tabular}{|c|c|c|c|c|}\n')
				# write column labels
				FID.write('\hline\n')
				FID.write('$T_s$ & $\mathrm{MSE}_1$ & $\mathrm{MSE}_2$ & Conservative fraction & Optimistic fraction \\\\\n')
				for k in range(3):
					FID.write('\hline\n')
					FID.write('%8.4g & %8.4g & %8.4g & %8.4g & %8.4g \\\\\n' % (TS[k],self.data[nb,k,0],self.data[nb,k,1],self.data[nb,k,2],self.data[nb,k,3]))
				FID.write('\\hline\n')
				FID.write('\end{tabular}\n')
				FID.write('\caption{Performance metrics for ')
				FID.write('%s ' % (self.name.upper()))
				if self.notes is not None:
					FID.write('(%s) ' % self.notes)
				if nb == 0:
					FID.write('with no forcing}\n')
				if nb == 1:
					FID.write('with Gaussian forcing term}\n')
				if nb == 2:
					FID.write('with cosine forcing term}\n')
				if nb == 3:
					FID.write('with Gaussian and cosine forcing terms}\n')
				## create figure label
				nam = 'tab:' + self.name + '_case_%d' % nb
				FID.write('\label{%s}\n' % nam)
				FID.write('\end{table}\n')
	## print a single table line with the metrics for comparing all filters at a constant sample rate
	def print_line_comparison(self,Ts_target,namebit):
		nind = namebit
		if Ts_target == 0.01:
			uind = 0
		elif Ts_target == 0.1:
			uind = 1
		elif Ts_target == 1.0:
			uind = 2
		lineOut = '%s & %8.4g & %8.4g & %8.4g & %8.4g \\\\\n' % (self.name.upper(),self.data[nind,uind,0],self.data[nind,uind,1],self.data[nind,uind,2],self.data[nind,uind,3])
		return lineOut

def main(argin='../trials'):
	# file format is "metric_<filter>_sims_<namebit>_<samplerate>.txt"
	filters = []
	for fn in os.listdir(argin):
		# check if name matches formal
		lf = len(fn)
		if lf > 6:
			metricsCheck = fn[0:7]
			extensionCheck = fn[(lf-3):(lf+1)]
			if metricsCheck == 'metrics' and extensionCheck == 'txt':
				fnlist = fn.split('_')
				filtername = fnlist[1]
				# if filtername == "sir", append the number of particles to the name
				if filtername == "sir":
					filtername = filtername + '(' + fnlist[2] + ')'
				juse = -1
				if len(filters) > 0:
					for j in range(len(filters)):
						if filtername == filters[j].name:
							juse = j
							break
				if juse < 0:
					# append
					filters.append(filterdat(filtername))
					juse = len(filters)-1
				print(filtername,filters[juse].name,juse)
				if filtername[0:3] == 'sir':
					Ns = int(fnlist[2])
					filters[juse].notes = '%d particles' % (Ns)
					namebit = int(fnlist[4],2)
					samplerate = (fnlist[5].split('.'))[0]
				else:
					namebit = int(fnlist[3],2)
					samplerate = (fnlist[4].split('.'))[0]
				print(samplerate)
				if samplerate == 'fast':
					Ts = 0.01
				elif samplerate == 'medium':
					Ts = 0.1
				elif samplerate == 'slow':
					Ts = 1.0
				print('Filter %s, namebit %d, Ts = %f' % (filtername,namebit,Ts))
				# set_data
				FID = open(argin+'/'+fn,'r')
				gar = FID.readline()
				print(gar)
				filters[juse].set_data(Ts,namebit,FID)
				FID.close()
	if len(filters)>0:
		FID = open('tex_tables.txt','w')
		for k in range(len(filters)):
			filters[k].print_out(FID)
		FID.close()
		# now, open each filter with the sample rate Ts = 0.1
		Tstar = [0.1,0.01,1.0]
		# and print a table for that case
		for i in range(len(Tstar)):
			FID = open('../../../estimationProjectWriteup/doc/tex/tex_comparison_tables_%d.tex' % (i),'w')
			for ko in range(1,4):
				FID.write('\n\n%%************************************************\n%% Performance table at Ts =  %f with namebit %d\n' % (Tstar[i],ko))
				# print to terminal
				FID.write('\\begin{table}[h!]\n')
				FID.write('\\centering\n')
				FID.write('\\begin{tabular}{|c|c|c|c|c|}\n')
				FID.write('\\hline\nFilter & $\mathrm{MSE}_1$ & $\mathrm{MSE}_2$ & Conservative fraction & Optimistic fraction \\\\\n')
				for k in range(len(filters)):
					FID.write('\hline\n')
					printline = filters[k].print_line_comparison(Tstar[i],ko)
					FID.write(printline)
				FID.write('\\hline\n')
				FID.write('\end{tabular}\n')
				FID.write('\caption{Performance metrics comparison of all filters with $T_s$ = %.2f sec and ' % (Tstar[i]))
				if ko == 0:
					FID.write('no forcing}\n')
				if ko == 1:
					FID.write('Gaussian forcing term}\n')
				if ko == 2:
					FID.write('cosine forcing term}\n')
				if ko == 3:
					FID.write('Gaussian and cosine forcing terms}\n')
				## create figure label
				nam = 'table:compare_case_%d_sample_%d' % (ko,i)
				FID.write('\label{%s}\n' % nam)
				FID.write('\end{table}\n')
			FID.close()
			print("Wrote to file tex_comparison_tables_%d.tex" % (i))




if __name__ == "__main__":
	main()
