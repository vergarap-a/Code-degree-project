import numpy as np
import time
import cv2
import pylab
import os
import sys


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[70,125], data_spike_limit=250,
                 face_detector_smoothness=10, input_age=21):

        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 15 #250
        #self.window = np.hamming(self.buffer_size)
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpmtimes = []
        self.frimattimes = []
        self.bpm = 0
        self.bpm_list = []
        self.bpm_limits = bpm_limits

        #FRIMAT VARIABLES
        self.fcm = []
        self.edad = float(input_age)
        self.fcmmax = 208 - 0.7*self.edad
        self.deltafc = []
        self.fcr = 60
        self.cca = 0
        self.ccr = []
        self.fmtfcm = 0
        self.fmtdeltafc = 0
        self.fmtfcmax = 0
        self.fmtccr = 0
        self.frimat_act = 0
        fmt = []
        self.fmt_value = 0
        self.fmt_list = []
        self.frimat_puntaje = []

        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)

        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13
        self.trained = False

        self.idx = 1
        self.find_faces = True

    #def find_faces_toggle(self):
        #self.find_faces = not self.find_faces
        #return self.find_faces

    def get_faces(self):
        return

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)

        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(0, 255, 0)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_means(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def train(self):
        self.trained = not self.trained
        return self.trained

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self, cam):
        self.times.append(time.time() - self.t0)

        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in,
                                                  cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)



        #Face detection - method
        # self.data_buffer, self.times, self.trained = [], [], False
        detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                            scaleFactor=1.3,
                                                            minNeighbors=4,
                                                            minSize=(
                                                                50, 50),
                                                            flags=cv2.CASCADE_SCALE_IMAGE))

        if len(detected) > 0:
            detected.sort(key=lambda a: a[-1] * a[-2])

            if self.shift(detected[-1]) > 10:
                self.face_rect = detected[-1]

        forehead1 = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
        self.draw_rect(self.face_rect, col=(255, 0, 0))
        x, y, w, h = self.face_rect
        cv2.putText(self.frame_out, "Face",
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        self.draw_rect(forehead1)
        x, y, w, h = forehead1
        cv2.putText(self.frame_out, "Forehead",
                    (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        
        
        if len(detected) == 0:
            self.times.pop(-1)
        
               
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        
        cv2.putText(
            #ACA ESTA 
            self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                cam),
            (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.25, col)
        cv2.putText(self.frame_out, "Press 'F' to save",
                   (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'E' to change age",
                   (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'N' to change level",
                   (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'R' to restart",
                   (10, 125), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

        if len(detected) > 0:
            vals = self.get_subface_means(forehead1)
            self.data_buffer.append(vals)
        
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > self.bpm_limits[0]) & (freqs < self.bpm_limits[1]))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            try:
                idx2 = np.argmax(pruned)
            except:
                raise ValueError()
            t = (np.sin(phase[idx2]) + 1.) / 2.
            t = 0.9 * t + 0.1
            alpha = t
            beta = 1 - t

            #Frequence of grater amplitude = heart rate!
            
            self.bpm = self.freqs[idx2]
            self.idx += 1

            if len(detected) > 0:
                self.bpm_list.append(int(self.bpm))
                self.bpmtimes.append((time.time()-self.t0))

            #Calculate FRIMAT

            #if len(self.bpm) > self.buffer_size and len(detected) > 0 :
            #self.fft      : frecuencia cardiaca 
            #self.FCR      : frecuencia cardiaca en reposo
            #self.FCM      : frecuencia cardiaca media en el instante actual
            #self.FCMmax   : frecuencia cardiaca en el percentil 95
            #self.deltaFC  : aceleracion de la FC
            #self.CCA      : FCM - FCR
            #self.CCR      : costo cardiaco relativo
            #self.FRIMAT   : valor total FRIMAT


            if len(self.bpm_list) > self.buffer_size and len(detected) > 0:
                self.fcm = np.mean(self.bpm)
                self.deltafc = self.fcmmax-self.fcm
                self.cca = self.fcm - self.fcr                  
                self.ccr = np.round((abs((self.cca)/(self.fcmmax-self.fcr))*100),0)

                fcm = [[1, 90, 94], [2, 95, 99], [4, 100, 104], [5, 105, 109]]
                value = self.fcm
                for i in range(len(fcm)):
                    rango = fcm[i]
                    if   rango[1] <= value and rango[2] >= value:
                        self.fmtfcm = rango[0]
                        break

                    elif value >= 110:
                        self.fmtfcm = 6
                        break
                    elif value < 90:
                        self.fmtfcm = 1
                        break
                
                deltafc = [[1, 20, 24], [2, 25, 29], [4, 30, 34], [5, 35, 39]]
                value = self.deltafc
                for i in range(len(fcm)):
                    rango = fcm[i]
                    if   rango[1] <= value and rango[2] >= value:
                        self.fmtdeltafc = rango[0]
                        break

                    elif value >= 40:
                        self.fmtdeltafc = 6
                        break               
                    elif value < 20:
                        self.fmtdeltafc = 1
                        break   
                fcmmax = [[1, 110, 119], [2, 120, 129], [4, 130, 139], [5, 140, 149]]

                value = self.fcmmax
                for i in range(len(fcm)):
                    rango = fcm[i]
                    if   rango[1] <= value and rango[2] >= value:
                        self.fmtfcmax = rango[0]
                        break

                    elif value >= 150:
                        self.fmtfcmax = 6
                        break               
                    elif value < 110:
                        self.fmtfcmax = 1
                        break   
                ccr = [[1, 10], [2, 15], [4, 20], [5, 25], [6, 30]]
                value = self.ccr
                dif = []
                for i in range(len(ccr)):
                    j = ccr[i]
                    dif.append(abs(value-j[1]))

                ccr_min = min(dif)
                id_ccr = dif.index(ccr_min)
                ccr_select = ccr[id_ccr]
                self.fmtccr = ccr_select[0]

                #frimat module

                self.frimat_act = self.fmtfcm + self.fmtfcmax + self.fmtdeltafc + self.fmtccr
                fmt = [["carga fisica min", 10],["muy ligero", 12],["ligero", 14],["soportable", 18],["penoso", 20],["duro", 22],["muy duro", 24],["extremadamente duro", 25]]
                value = self.frimat_act
                for i in range(len(fmt)):
                    j = fmt[i]
                    dif.append(abs(value-j[1]))

                fmt_min = min(dif)
                id_fmt = dif.index(ccr_min)
                fmt_select = fmt[id_fmt]
                self.fmt_value = fmt_select[0]
                self.frimattimes.append((time.time()-self.t0))
                self.fmt_list.append(self.fmt_value)
                self.frimat_puntaje.append(self.frimat_act)
                text2 = "Puntaje - " + str(self.frimat_act)
                text3 = "Categoria - " + str(self.fmt_value)
                
            x, y, w, h = self.get_subface_coord(0.5, 0.18, 0.25, 0.15)
            r = alpha * self.frame_in[y:y + h, x:x + w, 0]
            g = alpha * \
                self.frame_in[y:y + h, x:x + w, 1] + \
                beta * self.gray[y:y + h, x:x + w]
            b = alpha * self.frame_in[y:y + h, x:x + w, 2]
            self.frame_out[y:y + h, x:x + w] = cv2.merge([r,
                                                          g,
                                                          b])
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]
            col = (100, 255, 100)
            
            gap = (self.buffer_size - L) / self.fps
            # self.bpms.append(bpm)
            # self.ttimes.append(time.time())
            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)
            tsize = 1
            cv2.putText(self.frame_out, text,
                       (int(x - w / 2), int(y+h*1.4)), cv2.FONT_HERSHEY_PLAIN, tsize, col)

            if len(self.bpm_list) > self.buffer_size and len(detected) > 0:
                cv2.putText(self.frame_out, text2,
                        (int(x - w / 2), int(y+h*2.8)), cv2.FONT_HERSHEY_PLAIN, tsize, col) 
                cv2.putText(self.frame_out, text3,
                        (int(x - w / 2), int(y+h*4)), cv2.FONT_HERSHEY_PLAIN, tsize, col)               