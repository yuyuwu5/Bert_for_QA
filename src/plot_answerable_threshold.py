import matplotlib.pyplot as plt
import numpy as np
import json

RESULT1 = "./result_for_fig/result0.1.json"
RESULT3 = "./result_for_fig/result0.3.json"
RESULT5 = "./result_for_fig/result0.5.json"
RESULT7 = "./result_for_fig/result0.7.json"
RESULT9 = "./result_for_fig/result0.9.json"

def main():
	all_result = [RESULT1, RESULT3, RESULT5, RESULT7, RESULT9]
	exp = [0.1, 0.3, 0.5, 0.7, 0.9]
	answerable_em = []
	answerable_f1 = []
	unanswerable_em = []
	unanswerable_f1 = []
	overall_em = []
	overall_f1 = []
	for doc in all_result:
		with open(doc) as f:
			result = json.load(f)
		answerable_em.append(result['answerable']['em'])
		answerable_f1.append(result['answerable']['f1'])
		unanswerable_em.append(result['unanswerable']['em'])
		unanswerable_f1.append(result['unanswerable']['f1'])
		overall_em.append(result['overall']['em'])
		overall_f1.append(result['overall']['f1'])
	plt.subplot(1,2,2)
	plt.suptitle('Performance on Different Threshold')
	plt.title('EM')
	plt.plot(exp, answerable_em, '.-', label="answerable")
	plt.plot(exp, unanswerable_em, '.-', label="unanswerable")
	plt.plot(exp, overall_em, '.-', label="overall")
	plt.xticks(exp)
	plt.xlabel('answerable threshold')
	plt.legend(bbox_to_anchor=(1.45,1.3), loc='upper right',borderaxespad=3.)
	plt.subplot(1,2,1)
	plt.title('F1')
	plt.plot(exp, answerable_f1, '.-', label="answerable")
	plt.plot(exp, unanswerable_f1, '.-', label="unanswerable")
	plt.plot(exp, overall_f1, '.-', label="overall")
	plt.xlabel('answerable threshold')
	plt.xticks(exp)
	plt.tight_layout()
	plt.savefig("answerable_threshold.png")
	return

if __name__ == "__main__":
	main()
