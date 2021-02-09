#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.callbacks import CSVLogger

import csv
import numpy as np

from collections import OrderedDict
from collections import Iterable
from datetime import datetime

class CSVLoggerTimestamp(CSVLogger):
	def __init__(self, filename, separator=',', append=False ):
		super(CSVLoggerTimestamp, self).__init__(filename, separator, append)

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}

		def handle_value(k):
			is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
			if isinstance(k, Iterable) and not is_zero_dim_ndarray:
				return '"[%s]"' % (', '.join(map(str, k)))
			else:
				return k

		if not self.writer:
			self.keys = sorted(logs.keys())

			class CustomDialect(csv.excel):
				delimiter = self.sep

			self.writer = csv.DictWriter(self.csv_file, fieldnames=['timestamp'] + ['epoch'] + self.keys , dialect=CustomDialect)

			if self.append_header:
				self.writer.writeheader()

		timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
		row_dict = OrderedDict({'timestamp': timestamp})
		row_dict.update({'epoch': epoch})
		row_dict.update((key, handle_value(logs[key])) for key in self.keys)
		self.writer.writerow(row_dict)
		self.csv_file.flush()


