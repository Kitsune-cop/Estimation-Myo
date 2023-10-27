from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread

import myo
import numpy as np
import time
import quaternion


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)
    self.orientation = deque(maxlen=n)

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue), list(self.orientation)

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    with self.lock:
      self.emg_data_queue.append((event.timestamp, event.emg))

  def on_orientation(self, event):
    with self.lock:
      self.orientation.append((event.timestamp, event.orientation))

class Plot(object):

  def __init__(self, listener):
    self.n = listener.n
    self.listener = listener
    self.fig = plt.figure()
    self.axes = [self.fig.add_subplot(int('81' + str(i))) for i in range(1, 9)]
    [(ax.set_ylim([-100, 100])) for ax in self.axes]
    self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    plt.ion()
    self.count = 0

  def update_plot(self):
    emg_data,orientation = self.listener.get_emg_data()
    # print(orientation,type(emg_data))
    emg_data = np.array([x[1] for x in emg_data])
    orientation = np.array([quaternion.as_float_array(x[1]) for x in orientation])
    print(orientation)
    if len(emg_data) >= 200 and len(orientation) >= 200:
      print(emg_data.shape, orientation[:, None].shape)
      data = np.array(np.concatenate((emg_data,orientation[:, None]), axis=1))
      print(data)
      np.savetxt('./data_valo/neon_C_' + str(self.count) + '_.csv', data, delimiter=',')
      # np.savetxt('./data-valo/p000/04/p000_rock_' + str(self.count) + '_.csv', emg_data, delimiter=',')
      # np.savetxt('./data-valo/p000/04/p000_relax_' + str(self.count) + '_.csv', emg_data, delimiter=',')
      # np.savetxt('./data-valo/p000/04/p000_paper_' + str(self.count) + '_.csv', emg_data, delimiter=',')
      print(self.count)
      self.count += 1
    if self.count >= 60 :
      exit()
    emg_data = emg_data.T
    for g, data in zip(self.graphs, emg_data):
      if len(data) < self.n:
        # Fill the left side with zeroes.
        data = np.concatenate([np.zeros(self.n - len(data)), data])
      g.set_ydata(data)
      # print(len(data))
    plt.draw()

  def main(self):
    while True:
      # print('\r', count, end='')
      if self.count % 20 == 0 and self.count < 60:
        print(self.count, ' - ', self.count + 20)
        for i in range(3):
          print(3-i)
          time.sleep(1)
      self.update_plot()
      plt.pause(1.0)


def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(200)
  with hub.run_in_background(listener.on_event):
    Plot(listener).main()


if __name__ == '__main__':
  main()
