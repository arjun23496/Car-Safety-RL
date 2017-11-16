from interface import AEInterface

interface = AEInterface(number_of_trials=1, number_of_episodes=1000, horizon=500, test_horizon=100)

interface.execute(debug=True, persist=True, reload=False, mode=0, filepath='stest')