#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Fawang Zhang

#  Description: file for CarSim/TruckSim co-simulation
#  Update Date: 2024-03-24, Fawang Zhang: create example

import ctypes
import os
import sys

class VehicleSimulation:
    def __init__(self):
        self.dll_handle = None

    def get_api(self, dll_handle):
        self.dll_handle = dll_handle

        if dll_handle.vs_run is not None and \
                        dll_handle.vs_initialize is not None and \
                        dll_handle.vs_read_configuration is not None and \
                        dll_handle.vs_integrate_io is not None and \
                        dll_handle.vs_copy_export_vars is not None and \
                        dll_handle.vs_terminate_run is not None and \
                        dll_handle.vs_error_occurred is not None and \
                        dll_handle.vs_road_l is not None:
            dll_handle.vs_road_l.restype = ctypes.c_double
            #vsimports = (ctypes.c_double*1)()
            #vsexports = (ctypes.c_double*1)()
            #dll_handle.vs_integrate_io.argtypes = [ctypes.c_double, ctypes.byref(vsimports), ctypes.byref(vsexports)]
            ##dll_handle.vs.integrate_io.argtypes = [ctypes.c_double, ctypes.c_int]
            return True
        else:
            return False

    def get_char_pointer(self, python_string):
        # python version is greater or equal to 3.0 then we need to define the encoding when converting a string to
        # bytes. Once that is done we can convert the the python string to a char*.
        if sys.version_info >= (3, 0):
            char_pointer = ctypes.c_char_p(bytes(python_string, 'UTF-8'))
        else:
            char_pointer = ctypes.c_char_p(bytes(python_string))
        return char_pointer

    def get_parameter_value(self, line):
        index = line.find(' ')
        if index >= 0:
            return line[index:].strip()
        else:
            return None

    def get_dll_path(self, path_to_sim_file):
        dll_path = None
        prog_dir = None
        veh_code = None

        sim_file = open(path_to_sim_file, 'r')
        for line in sim_file:
            if line.lstrip().startswith('DLLFILE'):
                dll_path = self.get_parameter_value(line)
            elif line.lstrip().startswith('VEHICLE_CODE'):
                veh_code = self.get_parameter_value(line)
            elif line.lstrip().startswith('PROGDIR'):
                prog_dir = self.get_parameter_value(line)

        sim_file.close()

        if dll_path is None:
            dll_dir = 'Default64' if ctypes.sizeof(ctypes.c_voidp) == 8 else 'Default'
            dll_path = os.path.join(prog_dir, "Programs", "Solvers", dll_dir, veh_code + ".dll")
        return dll_path

    def run(self, path_to_sim_file):
        error_occurred = 1
        path_to_sim_file_ptr = self.get_char_pointer(path_to_sim_file)

        if path_to_sim_file_ptr is not None:
            error_occurred = self.dll_handle.vs_run(path_to_sim_file_ptr)

        return error_occurred

    def ReadConfiguration(self, path_to_sim_file):
        path_to_sim_file_ptr = self.get_char_pointer(path_to_sim_file)
        if path_to_sim_file_ptr is not None:
            ref_n_import = ctypes.c_int32()
            ref_n_export = ctypes.c_int32()
            ref_t_start = ctypes.c_double()
            ref_t_stop = ctypes.c_double()
            ref_t_step = ctypes.c_double()
            self.dll_handle.vs_read_configuration(path_to_sim_file_ptr,
                                                  ctypes.byref(ref_n_import),
                                                  ctypes.byref(ref_n_export),
                                                  ctypes.byref(ref_t_start),
                                                  ctypes.byref(ref_t_stop),
                                                  ctypes.byref(ref_t_step))
            configuration = {'n_import': ref_n_import.value,
                             'n_export': ref_n_export.value,
                             't_start': ref_t_start.value,
                             't_stop': ref_t_stop.value,
                             't_step': ref_t_step.value}
            return configuration

    def CopyExportVars(self, n_export):
        export_array = (ctypes.c_double * n_export)()
        self.dll_handle.vs_copy_export_vars(ctypes.cast(export_array, ctypes.POINTER(ctypes.c_double)))
        export_list = [export_array[i] for i in range(n_export)]
        return export_list

    def GetRoadL(self, x, y):
        x_c_double = ctypes.c_double(x)
        y_c_double = ctypes.c_double(y)
        c_double_return = self.dll_handle.vs_road_l(x_c_double, y_c_double)
        return float(c_double_return)

    def IntegrateIO(self, t_current, import_array, export_array):
        t_current_c_double = ctypes.c_double(t_current)
        import_c_double_array = (ctypes.c_double * len(import_array))(*import_array)
        export_c_double_array = (ctypes.c_double * len(export_array))(*export_array)

        c_integer_return = self.dll_handle.vs_integrate_io(t_current_c_double,
                                                           ctypes.byref(import_c_double_array),
                                                           ctypes.byref(export_c_double_array))

        export_array = [export_c_double_array[i] for i in range(len(export_array))]
        return c_integer_return, export_array

    def Initialize(self, t):
        t_c_double = ctypes.c_double(t)
        self.dll_handle.vs_initialize(t_c_double)

    def TerminateRun(self, t):
        t_c_double = ctypes.c_double(t)
        self.dll_handle.vs_terminate_run(t_c_double)