from .dotdict import dotdict


TrainArgs = dotdict({
	'LoadModel': False,
	'CheckPoint': './checkpoint/model.pth',
	'DataPath': './dataset/',
	'CudaDevice': '0',
	'subjects': [str(index).zfill(2) for index in range(30)],
	'actions': ['walk', 'wave', 'run', 'jump'],
	'GpePaths': ['1-person/01', '1-person/02', '2-person', '3-person', '4-person', '5-person'],
	'SpeTrainRatio': 0.3,
	'GpeTrainRatio': 0.8,

	'MaxEpoch': 30,
	'LearningRate': 0.001,
	'PrintFreq': 100,
	'BatchSize': 16,
	'k': 2,
	'milestone': [10, 15, 20, 25, 30, 35, 40, 45, 50],
	
	'NumFrames': 5,
	'DilatedRate': 3

})



TestArgs = dotdict({
	'CheckPoint': './checkpoint/model.pth',
	'DataPath': './dataset/',
	'CudaDevice': '0',
	'subjects': [str(index).zfill(2) for index in range(30)],

	# uniform distribution
	'SpeInfo': [('walk', [range(1000, 2000, 20)]),
				('wave', [range(1000, 2000, 20)]),
				('run', [range(500, 1000, 10)]),
				('jump', [range(500, 1000, 10)])],

	# range = (len(data)*0.9-250, len(data)*0.9+250)
	'GpeInfo': [('5-person', range(6400, 6900)),
				('4-person', range(18450, 18950)),
				('3-person', range(24700, 25200)),
				('2-person', range(16750, 17250)),
				('1-person/01', range(5700, 5950)),
				('1-person/02', range(5650, 5900))],
				
	'NumFrames': 5,
	'DilatedRate': 3

})