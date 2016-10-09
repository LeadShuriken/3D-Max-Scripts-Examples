import os
import datetime
import sys
import glob
import re
import pickle
from scipy import misc
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from power_law_fitting import *
import numpy as np

class LightSettings:
    
    def __init__(self, source_path):
        # Dictionary of data components
        #  'data' for max values vs settings
        #  'amp' amplitude curve fit parameter vs settings
        #  'index' index curve fit parameter vs settings
        self.data = {}
        self.source_path = source_path

        self.build_list(self.source_path)
        #self.fit_curve(self.source_path)
        #self.get_intensity_dependencies()
    #---
    # Class and override methods
    #---
    @classmethod
    def get_distance_max_values(cls, setting):
        # required offset 
        # index 0 => distance 1 meter
        xdata = [image[0] + 1 for image in setting]
        ydata = [image[1] for image in setting]
        return xdata, ydata

    def get_setting(self, file_name):
        pass

    #---
    # Save and load data methods
    #---
    def save_data(self, object, name):
        file_name = self.data_filename + '.' + name
        print 'LightSettings :: save_data :', file_name
        with open(file_name, 'wb') as output:
            pickle.dump(object, output, 1)

    def load_data(self, path, name):
        if not self.is_data_saved(path, name):
            return False
        file_name = self.data_filename + '.' + name
        print 'LightSettings :: load_data :', file_name
        with open(file_name, 'rb') as input:
            self.data[name] = pickle.load(input)
        return self.data[name] is not None

    def is_data_saved(self, absolute_path, name):
        self.data_filename = absolute_path.split('\\')[-1]
        return os.path.exists(self.data_filename + '.' + name)

    #---
    # Build maximum value list
    #---
    def build_list(self, absolute_path, overwrite=False):
        if not self.load_data(absolute_path, 'data') or overwrite:
            self.data['data'] = []
            for root, dirs, files in os.walk(absolute_path):
                if files:
                    Curve_value = []
                    Curve_value_count = 0
                    files1= glob.glob(os.path.join( root, "*.bmp"))
                    files2 = glob.glob(os.path.join( root, "*.png"))
                    files = files1 + files2

                    if len(files) > 0:
                        set_list = []
                        for file in files:
                            max_value = self.get_max(file)
                            dist, intensity, range, directionality = self.get_setting(file)
                            set_list.append((float(dist), float(max_value), float(intensity), float(range), float(directionality)))

                        set_list = sorted(set_list, key=lambda sort_key: sort_key[0])
                        self.data['data'].append(set_list)
            print 'build_list : number of settings :', len(self.data['data'])
            self.save_data(self.data['data'], 'data')

    def get_max(self, full_path_name):

        # Get Resolution ---
        img = misc.imread(full_path_name)
        img_resolution = img.shape[0]
        offset_x = str((img_resolution/2) - 20)
        offset_y = str((img_resolution/2) - 10)

        converted_res = "temp.txt"
        command_string = 'convert -colorspace Gray -crop 20x20+' + offset_x + '+' + offset_y + ' -verbose -identify ' + full_path_name + ' nul 2>&1> ' + converted_res
        results = os.system(command_string)

        with open(converted_res , 'r') as f:
            line = f.readline()
            while len(line) > 0:
                if line.find('max:') > -1:
                    max_value = line.split()[1]
                    break
                line = f.readline()
        
        return max_value

    #---
    # Build curve fit parameter list
    #---
    def fit_curve(self, absolute_path, overwrite=False):
        if not self.load_data(absolute_path, 'amp') or not self.load_data(absolute_path, 'index') or overwrite:
            print 'fit_curve :: number of settings :', len(self.data['data'])
            amp_list = []
            index_list = []
            for setting in self.data['data']:
                print 'fit_curve :: setting i:%s r:%s' % (setting[0][2], setting[0][3])
                xdata, ydata = self.get_distance_max_values(setting)
                amp, index = power_law_curve_fit(xdata, ydata, True)
                if amp is None:
                    print 'ERR :: fit_curve :: no solution for intensity %s range %s' % (setting[0][2], setting[0][3])
                amp_list.append(amp)
                index_list.append(index)
            self.data['amp'] = amp_list
            self.data['index'] = index_list
            self.save_data(self.data['amp'], 'amp')
            self.save_data(self.data['index'], 'index')
        print 'fit_curve :: first fit amp:%f index:%f', (self.data['amp'][0], self.data['index'][0])

    def get_intensity_dependencies(self):
        # Collect intensity/lumen and powerlaw function parameters
        xdata = [setting[0][2] for setting in self.data['data']]
        ydata1 = self.data['amp']
        ydata2 = self.data['index']
        
        # Sort on lumen 
        data_zipped = zip(xdata, ydata1, ydata2)
        data_zipped = sorted(data_zipped, key=lambda sort_key: sort_key[0])
        xdata = [x[0] for x in data_zipped]
        ydata1 = [x[1] for x in data_zipped]
        ydata2 = [x[2] for x in data_zipped]
        
        value = get_slope(xdata, ydata1, ydata2, False)

    def plot(self):
        for setting in self.data['data']:
            xdata = [image[0] for image in setting]
            ydata = [image[1] for image in setting]
            
            clf()
            plot(xdata, ydata)     # Fit
            raw_input("Press Enter to continue...")

    #---
    # Raw data match functions
    #---
    def create_combinations(self, absolute_path, overwrite=False):
        if not self.load_data(absolute_path, 'combination') or overwrite:
            self.data['combination'] = []
            n_settings = len(self.data['data'])
            for first in range(0, n_settings):
                for second in range(first, n_settings):
                    print 'create_combinations :: combine %d with %d' % (first, second)
                    first_max_list = np.array(self.data['data'][first], dtype=np.float)
                    second_max_list = np.array(self.data['data'][second], dtype=np.float)
                    first_max_list = first_max_list[:, 1]
                    second_max_list = second_max_list[:, 1]
                    curve = np.add(first_max_list, second_max_list)
                    too_big_ind = curve > 255
                    curve[too_big_ind] = 255
                    first_intensity = self.data['data'][first][0][2]
                    first_range = self.data['data'][first][0][3]
                    second_intensity = self.data['data'][second][0][2]
                    second_range = self.data['data'][second][0][3]

                    combination = [first_intensity, first_range, second_intensity, second_range, curve]
                    self.data['combination'].append(combination)
            self.save_data(self.data['combination'], 'combination')

    def euler_curve_match(self, absolute_path, settings, overwrite=False, plot_results=False):
        if not self.load_data(absolute_path, 'match') or overwrite:
            n_curve_points = len(self.data['data'][0])
            self.data['match'] = []
            for setting in self.data['data']:
                curve = np.array(setting, dtype=np.float)[:, 1]
                error = []
                for combi in settings.data['combination']:
                    combi_curve = combi[4]
                    curve_diff = np.power(np.subtract(curve, combi_curve), 2)
                    error.append(np.sqrt(np.sum(curve_diff)))

                best_match_index = np.argmin(error)
                best_match_settings = settings.data['combination'][best_match_index]
                self.data['match'].append(best_match_settings)
            self.save_data(self.data['match'], 'match')

        if plot_results:
            #self.plot_curve_match()
            #self.plot_settings_conversion()
            #self.plot_settings_conversion_angle()
            self.plot_settings_conversion_angle_3d()

    def plot_per_setting(self, settings):
        for setting in self.data['data']:
            lumen = setting[0][2]
            distance = np.array(setting, dtype=np.float)[:, 0]
            ref_curve = np.array(setting, dtype=np.float)[:, 1]

            clf()

            for setting2 in settings.data['data']:
                ref_curve2 = np.array(setting2, dtype=np.float)[:, 1]
                plot(distance, ref_curve2, 'b')

            plot(distance, ref_curve, 'r')
            title(lumen)
            xlabel('distance')
            ylabel('max')

            raw_input("Press Enter to continue...")

    def plot_curve_match(self):
        for (setting, match) in zip(self.data['data'], self.data['match']):
            lumen = setting[0][2]
            distance = np.array(setting, dtype=np.float)[:, 0]
            ref_curve = np.array(setting, dtype=np.float)[:, 1]
            match_curve = match[4]

            clf()
            h = subplot(1, 1, 1)
            plot(distance, ref_curve, 'r')
            plot(distance, match_curve, 'b')
            title(lumen)
            xlabel('distance')
            ylabel('max')

            h.set_xlim([0, 30])
            h.set_ylim([0, 260])
            print match[0:4]

            raw_input("Press Enter to continue...")

    def plot_settings_conversion(self):

        lumen = [setting[0][2] for setting in self.data['data']]

        intensity_1 = [match[0] for match in self.data['match']]
        intensity_2 = [match[2] for match in self.data['match']]
        range_1 = [match[1] for match in self.data['match']]
        range_2 = [match[3] for match in self.data['match']]

        clf()
        h = subplot(2, 1, 1)
        plot(lumen, intensity_1, 'r.')
        plot(lumen, intensity_2, 'b.')
        xlabel('lumen')
        ylabel('intensity')

        h = subplot(2, 1, 2)
        plot(lumen, range_1, 'r.')
        plot(lumen, range_2, 'b.')
        xlabel('lumen')
        ylabel('range')

        raw_input("Press Enter to continue...")

    def plot_settings_conversion_angle(self):

        angle = np.array([setting[0][4] for setting in self.data['data']])
        lumen = np.array([setting[0][2] for setting in self.data['data']])
        unique_lumen_values = set(lumen)

        intensity_1 = np.array([match[0] for match in self.data['match']])
        intensity_2 = np.array([match[2] for match in self.data['match']])
        range_1 = np.array([match[1] for match in self.data['match']])
        range_2 = np.array([match[3] for match in self.data['match']])

        ind = intensity_1 < intensity_2
        intensity_1[ind], intensity_2[ind] = intensity_2[ind], intensity_1[ind]
        range_1[ind], range_2[ind] = range_2[ind], range_1[ind]

        for lum in unique_lumen_values:
            ind = lumen == lum

            angle_for_lum = angle[ind]
            intensity_1_for_lum = intensity_1[ind]
            intensity_2_for_lum = intensity_2[ind]
            range_1_for_lum = range_1[ind]
            range_2_for_lum = range_2[ind]
            lumen_for_lum = lumen[ind]

            clf()
            h = subplot(2, 1, 1)
            plot(angle_for_lum, intensity_1_for_lum, 'r.')
            plot(angle_for_lum, intensity_2_for_lum, 'b.')
            xlabel('angle')
            ylabel('intensity')
            title('@lumen: %.0f' % lum)

            h.set_ylim([0, 2])

            h = subplot(2, 1, 2)
            plot(angle_for_lum, range_1_for_lum, 'r.')
            plot(angle_for_lum, range_2_for_lum, 'b.')
            xlabel('angle')
            ylabel('range')

            h.set_ylim([0, 150])

            raw_input("Press Enter to continue...")

    def plot_settings_conversion_angle_3d(self):

        angles = np.array([setting[0][4] for setting in self.data['data']])
        lumen = np.array([setting[0][2] for setting in self.data['data']])

        intensity_1 = np.array([match[0] for match in self.data['match']])
        intensity_2 = np.array([match[2] for match in self.data['match']])
        range_1 = np.array([match[1] for match in self.data['match']])
        range_2 = np.array([match[3] for match in self.data['match']])

        ind = intensity_1 < intensity_2
        intensity_1[ind], intensity_2[ind] = intensity_2[ind], intensity_1[ind]
        range_1[ind], range_2[ind] = range_2[ind], range_1[ind]
        unique_lumen = sorted(set(lumen))
        unique_angles = sorted(set(angles))

        fig = plt.figure()

        X, Y = np.meshgrid(unique_lumen, unique_angles)

        Z_i1 = np.zeros(X.shape)
        Z_i2 = np.zeros(X.shape)
        Z_r1 = np.zeros(X.shape)
        Z_r2 = np.zeros(X.shape)

        for i_lum, lum in enumerate(unique_lumen):
            for i_ang, ang in enumerate(unique_angles):
                ind = np.logical_and(lumen == lum, angles == ang)

                if np.any(ind):
                    Z_i1[i_ang, i_lum] = intensity_1[ind][0]
                    Z_i2[i_ang, i_lum] = intensity_2[ind][0]
                    Z_r1[i_ang, i_lum] = range_1[ind][0]
                    Z_r2[i_ang, i_lum] = range_2[ind][0]
                else:
                    print 'Missing Coordinates: ' ,unique_lumen[i_lum], unique_angles[i_ang]

        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot_surface(X, Y, Z_i1)
        ax.set_xlabel('lumen')
        ax.set_ylabel('angle')
        ax.set_zlabel('intensity1')

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.plot_surface(X , Y, Z_r1)
        ax.set_xlabel('lumen')
        ax.set_ylabel('angle')
        ax.set_zlabel('range1')

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.plot_surface(X , Y, Z_i2)
        ax.set_xlabel('lumen')
        ax.set_ylabel('angle')
        ax.set_zlabel('intensity2')

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.plot_surface(X , Y, Z_r2)
        ax.set_xlabel('lumen')
        ax.set_ylabel('angle')
        ax.set_zlabel('range2')

        title('')

        raw_input("Press Enter to continue...")


    def linear_fit_match(self, unity_settings, plot_results = False):

        lumen = np.array([setting[0][2] for setting in self.data['data']], dtype=np.float)

        intensity_1 = np.array([match[0] for match in self.data['match']], dtype=np.float)
        intensity_2 = np.array([match[2] for match in self.data['match']], dtype=np.float)
        range_1 = np.array([match[1] for match in self.data['match']], dtype=np.float)
        range_2 = np.array([match[3] for match in self.data['match']], dtype=np.float)

        # Sorting all data according to lumen
        sorted_lumen_ind = lumen.argsort()

        lumen = lumen[sorted_lumen_ind]
        intensity_1 = intensity_1[sorted_lumen_ind]
        intensity_2 = intensity_2[sorted_lumen_ind]
        range_1 = range_1[sorted_lumen_ind]
        range_2 = range_2[sorted_lumen_ind]

        # Trimming all data in the beginning and the end to exclude outliers
        valid_data = (lumen >= 1000) * (lumen <= 9000)
        lumen = lumen[valid_data]
        intensity_1 = intensity_1[valid_data]
        intensity_2 = intensity_2[valid_data]
        range_1 = range_1[valid_data]
        range_2 = range_2[valid_data]

        # Swap array outliers between sets (sometimes pair are formed but are logically swapped)
        i2_bigger_ind = intensity_2 > intensity_1
        i1 = intensity_1[i2_bigger_ind]
        i2 = intensity_2[i2_bigger_ind]
        r1 = range_1[i2_bigger_ind]
        r2 = range_2[i2_bigger_ind]
        intensity_1[i2_bigger_ind], intensity_2[i2_bigger_ind] = i2, i1
        range_1[i2_bigger_ind], range_2[i2_bigger_ind] = r2, r1

        # Line fit intensity assumed constant
        offset_intens_1 = intensity_1.mean()
        std_i1 =  intensity_1.std()
        offset_intens_2 = intensity_2.mean()
        std_i2 =  intensity_2.std()

        p1, residuals1, rank1, singular_values1, rcond1 = np.polyfit(lumen, range_1, 1, full=True)
        p2, residuals2, rank2, singular_values2, rcond2 = np.polyfit(lumen, range_2, 1, full=True)
        std_r1 = residuals1 #/ len(range_1)
        std_r2 = residuals2 #/ len(range_2)

        # present
        print 'LightSettings :: linear_fit_match : intensity 1 setting : i %1.2f , std %1.2e' % (offset_intens_1, std_i1)
        print 'LightSettings :: linear_fit_match : intensity 2 setting : i %1.2f , std %1.2e' % (offset_intens_2, std_i2)
        print 'LightSettings :: linear_fit_match : range 1 setting : lumen * %1.2e + %1.2e , std %1.2e' % (p1[0], p1[1], std_r1)
        print 'LightSettings :: linear_fit_match : range 2 setting : lumen * %1.2e + %1.2e , std %1.2e' % (p2[0], p2[1], std_r2)

        self.data['conversion'] = [('intensity_1', 0, offset_intens_1),
                                   ('range_1', p1[0], p1[1]),
                                   ('intensity_2', 0, offset_intens_2),
                                   ('range_2', p2[0], p2[1])]

        print self.data['conversion']

        if plot_results:
            self.plot_conversion_func()

    def plot_conversion_func(self):

        conversion = self.data['conversion']

        lumen = np.array([setting[0][2] for setting in self.data['data']], dtype=np.float)

        intensity_1 = np.array([match[0] for match in self.data['match']], dtype=np.float)
        intensity_2 = np.array([match[2] for match in self.data['match']], dtype=np.float)
        range_1 = np.array([match[1] for match in self.data['match']], dtype=np.float)
        range_2 = np.array([match[3] for match in self.data['match']], dtype=np.float)

        converted_int_1 = (lumen * conversion[0][1] + conversion[0][2])
        converted_range_1 = (lumen * conversion[1][1] + conversion[1][2])
        converted_int_2 = (lumen * conversion[2][1] + conversion[2][2])
        converted_range_2 = (lumen * conversion[3][1] + conversion[3][2])

        clf()

        h = subplot(2, 1, 1)
        plot(lumen, intensity_1, 'r.')
        plot(lumen, intensity_2, 'b.')
        plot(lumen, converted_int_1, 'g')
        plot(lumen, converted_int_2, 'g')
        xlabel('lumen')
        ylabel('intensity')

        h = subplot(2, 1, 2)
        plot(lumen, range_1, 'r.')
        plot(lumen, range_2, 'b.')
        plot(lumen, converted_range_1, 'g')
        plot(lumen, converted_range_2, 'g')
        xlabel('lumen')
        ylabel('range')

        raw_input("Press Enter to continue...")

    #---
    # Match settings by finding the smallest error of amps and indices
    #---
    def euler_match(self, settings, plot_results=False):
        # prepare data for numpy
        my_amp = np.array(self.data['amp'], dtype=np.float)
        my_index = np.array(self.data['index'], dtype=np.float)
        n_my_settings = my_amp.shape[0]

        their_amp = np.array(settings.data['amp'], dtype=np.float)
        their_index = np.array(settings.data['index'], dtype=np.float)
        n_their_settings = their_amp.shape[0]

        # 'reshape' for vectorisation
        my_amp = np.tile(my_amp, [n_their_settings, 1]).transpose()
        my_index = np.tile(my_index, [n_their_settings, 1]).transpose()
        their_amp = np.tile(their_amp, [n_my_settings, 1])
        their_index = np.tile(their_index, [n_my_settings, 1])

        # calculate difference
        amp_diff = ((my_amp - their_amp) ** 2) / my_amp ** 2
        index_diff = ((my_index - their_index) ** 2) / my_index ** 2
        weight = my_amp / my_index
        #weight = 4000
        err_distance = np.sqrt(amp_diff + weight * index_diff)

        # Determine best match with smallest error distance
        min_dist = np.nanmin(err_distance, 1)
        min_i = np.nanargmin(err_distance, 1)
        data = np.array(settings.data['data'], float)
        their_intensity = data[:, 0, 2]
        their_range = data[:, 0, 3]
        self.data['matched_intensity'] = their_intensity[min_i]
        self.data['matched_range'] = their_range[min_i]

        if plot_results:
            self.plot_match()

    def plot_match(self):
        data = np.array(self.data['data'], float)
        intensity = data[:, 0, 2]
        matched_intensity = self.data['matched_intensity']
        matched_range = self.data['matched_range']

        clf()
        subplot(211)
        plot(intensity, matched_intensity, 'o')
        xlabel('lumen')
        ylabel('intensity')

        subplot(212)
        plot(intensity, matched_range, 'o')
        xlabel('lumen')
        ylabel('range')

        raw_input("Press Enter to continue...")

class MaxSettings (LightSettings): 
        
    def get_setting (self, file_name):
        m = file_name.split('\\')[-1].split('_')
        lumen = m[1]
        dist = m[2]
        return (dist, lumen, '0', '0')
        
    def get_distance_max_values(self, setting):
        xdata, ydata = LightSettings.get_distance_max_values(setting)
        data_zipped = [value for value in zip(xdata, ydata) if value[1] > 0]
        xdata = [x[0] for x in data_zipped]
        ydata = [x[1] for x in data_zipped]
        return (xdata, ydata)

class MaxSettingsSpot (MaxSettings, LightSettings):

    def get_setting (self, file_name):
        m = file_name.split('\\')[-1].split('_')
        lumen = m[1]
        dist = m[2]
        directionality = m[4].split('.')[0]
        return (dist, lumen, '0', directionality)

    def get_distance_max_values(self, setting):
        xdata, ydata = LightSettings.get_distance_max_values(setting)
        data_zipped = [value for value in zip(xdata, ydata) if value[1] > 0]
        xdata = [x[0] for x in data_zipped]
        ydata = [x[1] for x in data_zipped]
        return (xdata, ydata)

class UnitySettings (LightSettings): 
        
    def get_setting (self, file_name):
        m = file_name.split("\\")[-1].split("_")
        dist = m[1]
        range = m[3]
        intensity = m[5]
        return (dist, intensity, range, '0')
    
    def get_distance_max_values(self, setting):
        xdata, ydata = LightSettings.get_distance_max_values(setting)
        data_zipped = [value for value in zip(xdata, ydata) if value[1] > 0]
        xdata = [x[0] for x in data_zipped]
        ydata = [x[1] for x in data_zipped]
        '''
        if len(ydata) > 1 and ydata[0] == ydata[1]:
            xdata = xdata[1:]
            ydata = ydata[1:]
        '''
        return (xdata, ydata)

if __name__ == '__main__':
    if len(sys.argv) > 2:
        max_absolute_path = str(sys.argv[1])
        unity_absolute_path = str(sys.argv[2])
    else:
        max_absolute_path = '\\\\code1\\storage\\2012-9114_OSS\\_sandbox\\IntensityAdjustment\\MaxRenders_360'
        max_spot_absolute_path = 'O:\\_sandbox\\IntensityAdjustment\\Max_allAngles_200and2000'

        unity_absolute_path = 'C:\\ossact\\max_unity_light_mapping\\UnityRenderers'

    unity_settings = UnitySettings(unity_absolute_path)
    unity_settings.create_combinations(unity_absolute_path)

    #max_settings = MaxSettings(max_absolute_path)
    #max_settings.euler_curve_match(max_absolute_path, unity_settings, False, True)
    #max_settings.linear_fit_match(unity_settings , True)

    max_spot_settings = MaxSettingsSpot(max_spot_absolute_path)
    max_spot_settings.euler_curve_match(max_spot_absolute_path, unity_settings, False, True)