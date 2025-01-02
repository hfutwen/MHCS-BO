from artiq.experiment import *
import numpy as np
import os
import h5py
import time


def RefFreq(delta, AOM, divisor) -> TFloat:
    # Delta = 6834.682611  # F1->F2频差(MHz)
    # F=495.8              # F'0->F'3频差(MHz)
    # AOM_cooling = 81.0  # 冷却光AOM移频
    # return np.round((Delta-F+AOM_cooling+delta)/divisor, 2)

    Delta = 6834.682611  # F1->F2频差(MHz)
    F = 459.706  # F'01->F'3频差(MHz)459.706
    # AOM = 80.0  # 冷却光AOM移频
    res = np.round((Delta - F + AOM + delta) / divisor, 3)
    return res


class Raman_Rabi(EnvExperiment):

    def build(self):
        # =====================================================================
        #         初始化所有device，对应device_db中的dict.key()
        # =====================================================================

        # core
        self.setattr_device("core")

        # leds
        self.setattr_device("led0")
        self.setattr_device("led1")

        # ttl ttl0-ttl3 are ttlInOut  ttl4-ttl15 are ttlOut
        for i in range(16):
            ttl_dev = "ttl" + str(i)
            self.setattr_device(ttl_dev)

        # urukul_cpld
        self.setattr_device("urukul0_cpld")

        # urukuls
        self.setattr_device("urukul0_ch0")
        self.setattr_device("urukul0_ch1")
        self.setattr_device("urukul0_ch2")
        self.setattr_device("urukul0_ch3")

        self.dds_ref = self.urukul0_ch0
        self.dds_cooling = self.urukul0_ch2
        self.dds_repump = self.urukul0_ch3

        self.ttl_ccd = self.ttl14
        self.ttl_magnet = self.ttl5
        self.ttl_sample = self.ttl10
        self.ttl_repump = self.ttl15
        self.ttl_Raman = self.ttl7
        self.ttl_lattice = self.ttl12
        self.ttl_detect = self.ttl13
        self.ttl_bias = self.ttl6
        self.ttl_guide = self.ttl9

        # samplers
        self.setattr_device("sampler0")

        # zotions
        self.setattr_device("zotino0")

        # 自定义的外部输入参数
        self.setattr_argument("deltaPGC", NumberValue(default=188.21))  # 219.12
        self.setattr_argument("deltaStep", NumberValue(default=67))  # 67
        self.setattr_argument("att2", NumberValue(default=31.5))  # 
        self.setattr_argument("att2_repump", NumberValue(default=31.5))  # 
        self.setattr_argument("attStep", NumberValue(default=50))  # 50
        self.setattr_argument("magDelay", NumberValue(default=6.87))  # 20

        self.setattr_argument("motAtLast", BooleanValue(default=0))  # PGC之后是否运行mot
        
        # 晶格相关
        self.setattr_argument("lattDelay", NumberValue(default=5.92))  # 晶格装载 lattDelay1 ms后打开磁场
        self.setattr_argument("tDa", NumberValue(default=0.17))         # 绝热过程持续时间，ms
        self.setattr_argument("Von", NumberValue(default=5))         # AOM射频信号调制电压最高点, V

        #晶格分段卸载
        self.setattr_argument("Vmid1", NumberValue(default=-0.))      
        self.setattr_argument("Vmid2", NumberValue(default=-2.0)) 
        self.setattr_argument("unload_n1", NumberValue(default=20)) 
        self.setattr_argument("unload_n2", NumberValue(default=50)) 
        self.setattr_argument("unload_n3", NumberValue(default=50)) 
        self.setattr_argument("holding1", NumberValue(default=5)) 
        self.setattr_argument("holding2", NumberValue(default=10)) 

        # 分段PGC相关
        self.setattr_argument("deltaT1", NumberValue(default=10))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT2", NumberValue(default=6.42))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT3", NumberValue(default=2.37))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT4", NumberValue(default=5.25))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT5", NumberValue(default=0.9))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT6", NumberValue(default=0.42))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT7", NumberValue(default=9.96))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT8", NumberValue(default=7.76))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT9", NumberValue(default=1.04))  # pgc截止时刻对应的冷却光AOM频率
        self.setattr_argument("deltaT10", NumberValue(default=0.3))  # pgc截止时刻对应的冷却光AOM频率


        self.deltaPGC = self.deltaPGC
        self.deltaStep = int(self.deltaStep)
        self.att2 = self.att2
        self.att2_repump = self.att2_repump
        self.attStep = int(self.attStep)
        self.motAtLast = self.motAtLast
        
        self.deltaT1 = self.deltaT1
        self.deltaT2 = self.deltaT2
        self.deltaT3 = self.deltaT3
        self.deltaT4 = self.deltaT4
        self.deltaT5 = self.deltaT5
        self.deltaT6 = self.deltaT6
        self.deltaT7 = self.deltaT7
        self.deltaT8 = self.deltaT8
        self.deltaT9 = self.deltaT9
        self.deltaT10 = self.deltaT10

    def prepare(self):
        # 构造PGC阶段所需功率序列
        deltaMOT = 19.5  # MOT失谐
        deltaPGC = self.deltaPGC 
        self.att1 = 17
        
        # 光晶格参数
        tDa = self.tDa    #绝热变化时间
        deltat = 20  # 绝热变化的时间步长，us，这是由时序控制器的DA决定的
        self.nAd = int(tDa * 1e3 / deltat)# 绝热过程的步数
        self.Von = self.Von    #光晶格常开时的调制电压，V
        self.Voff = -6.0 # 光晶格完全关闭时的调制电压，V
        self.deltaV = (self.Von - self.Voff) / self.nAd  #

        # 光晶格分段卸载, DA输出间隔为10us
        self.Vmid1 = self.Vmid1
        self.Vmid2 = self.Vmid2

        self.unload_n1 = int(self.unload_n1)
        self.unload_n2 = int(self.unload_n2)
        self.unload_n3 = int(self.unload_n3)

        self.deltaV1 = (self.Von - self.Vmid1) / self.unload_n1
        self.deltaV2 = (self.Vmid1 - self.Vmid2) / self.unload_n2
        self.deltaV3 = (self.Vmid2 - self.Voff) / self.unload_n3

        self.holding1 = self.holding1
        self.holding2 = self.holding2


        # 锁频参数
        self.cooling_MOT_Freq = RefFreq(deltaMOT, 80, 32.0)  # MOT冷却光参考频率
        self.cooling_PGC_Freq = RefFreq(deltaPGC, 80, 32.0)  # 根据失谐计算出冷却光参考频率参考频率
        self.detect_Freq = RefFreq(0.0, 80, 32.0)  # 根据失谐计算出探测光参考频率
        self.detect_Freq1 = RefFreq(5, 80, 32.0)  # 根据失谐计算出探测光参考频率
        self.TOF_detect_freq = RefFreq(0, 250, 32.0)  # 根据失谐计算出TOF探测光参考频率

        # 分段PGC用的参数
        self.deltaT = np.array([self.deltaT1, self.deltaT2, self.deltaT3, self.deltaT4, self.deltaT5,
                                self.deltaT6, self.deltaT7, self.deltaT8, self.deltaT9, self.deltaT10])
        f1 = self.cooling_MOT_Freq
        f2 = self.cooling_PGC_Freq
        N=len(self.deltaT)
        self.deltaFreq = (f2-f1)/N
        self.deltaAtt = (self.att2-self.att1)/N
        self.deltaAtt_repump = (self.att2_repump-19)/N


    @kernel
    def initialize(self):
        self.core.reset()
        delay(10 * ms)
        self.core.break_realtime()
        delay(100 * us)

        # self.dds_ref.init()
        self.dds_repump.set_att(25. * dB)  # 设置为18时，经功放后输出31.2dBm
        self.dds_repump.set(frequency=151 * MHz, phase=0.50)
        self.dds_repump.sw.on()

        self.ttl_detect.off()  # tof光开关
        self.ttl_magnet.off()  # 磁场开关
        self.ttl_ccd.off()  # ccd触发开关
        self.ttl_guide.off()
        self.ttl_Raman.off()
        self.ttl_bias.off()
        delay(10 * us)
        # 初始化DA及使用DA
        self.zotino0.init()#
        delay(500 * us)
        
    @kernel
    def run_MOT(self, loadTime):
        self.ttl_magnet.off()  # 磁场开关
        self.ttl_ccd.off()  # ccd触发开关

        delay(10 * us)

        # 开cooling光(探测光) 参考频率
        self.dds_ref.set(self.cooling_MOT_Freq * MHz)  # cooling光 参考信号频率
        self.dds_ref.set_att(2.0 * dB)  # cooling光参考信号功率
        self.dds_ref.sw.on()  # cooling光参考信号 开关:1-开、0-关
        delay(10 * us)

        # 开repump光 AOM
        self.ttl_repump.on()  # repump光开关
        delay(10 * us)

        # 开cooling光 AOM
        self.dds_cooling.set(frequency=80.0 * MHz, phase=0.0)
        self.dds_cooling.set_att(self.att1 * dB)
        self.dds_cooling.sw.on()

        self.ttl_magnet.on()  # 磁场开关
        delay(loadTime * ms)  # 装载时间
        

    # 光晶格绝热加载
    @kernel
    def lattice_upload(self):
        self.ttl_lattice.on()
        for j in range(self.nAd):
            Voltage = self.Voff + self.deltaV * (j + 1)
            self.zotino0.write_dac(0, Voltage)  
            delay(20 * us)  #
            self.zotino0.load()
            delay(80 * us)

    # 光晶格绝热卸载
    @kernel
    def lattice_unload(self):
        for j in range(self.nAd):
            Voltage = self.Von - self.deltaV * (j + 1)
            self.zotino0.write_dac(0, Voltage)  
            delay(10 * us)  #
            self.zotino0.load()
            delay(10 * us)
        self.ttl_lattice.off()

    
    @kernel
    def runMultiPGC(self):
        self.ttl_magnet.off()
        delay(self.magDelay*ms)

        for i in range(len(self.deltaT)):
            f = self.cooling_MOT_Freq + self.deltaFreq*i
            att = self.att1 + self.deltaAtt*i
            # att_repump = 19 + self.deltaAtt_repump*i

            self.dds_cooling.set_att(att * dB)
            # self.dds_repump.set_att(att_repump * dB)

            self.dds_ref.set(frequency=f * MHz)

            delay(self.deltaT[i] * ms)


        self.dds_cooling.sw.off()
        delay(1*ms)
        self.ttl_repump.off()

        


    @kernel
    def detectFluPulse(self,hoding,t):
        # self.dds_ref.set(207.031 * MHz) # TOF探测光的参考频率
        self.dds_ref.set(frequency=self.TOF_detect_freq * MHz, phase=0.5)  # TOF探测光的参考频率
        self.ttl_repump.off()
        # delay(200 * us)
        # TOF测温

        delay((220) * ms)
        self.ttl_detect.on()
        self.ttl_sample.pulse(50 * us)

        delay(200*ms)
        self.ttl_detect.off()

    @kernel
    def detectFlu(self):
        self.ttl_magnet.off()
        delay(10 * us)
        self.dds_cooling.sw.off()
        delay(10 * us)
        self.ttl_repump.off()
        self.dds_ref.set(self.detect_Freq1 * MHz)  # cooling光 参考信号频率，对应冷却光失谐
        
        self.dds_cooling.set_att(20.0 * dB)
        delay(10 * us)
        # self.ttl_detect.on()
        # self.ttl_repump.on()

        # delay(self.TOF * ms)  # TOF时间

        self.ttl_ccd.pulse(50 * us)
        self.dds_cooling.sw.on()
        self.ttl_repump.on()

        delay(200 * ms)
        self.ttl_ccd.pulse(50 * us)

        delay(100 * ms)
        self.dds_cooling.sw.off()
        self.ttl_repump.off()

    @kernel
    def run(self):
        self.initialize()

        # self.zotino0.write_dac(0, 5)  # 最大4V
        # delay(20 * us)  #
        # self.zotino0.load()
        # delay(30 * us)

        idleTime = 20
        hoding = 15  # 导引光作用时间, ms
        t = 20  # 探测延时, ms
        loading = 2000  # MOT装载时间,ms
        ramanPulse = range(0, 502, 2)
        # for i in range(len(ramanPulse) + idleTime):
        # for i in range(len(hoding)):
        for i in range(1):
        # for i in range(len(ramanPulse)):
            delay(10 * ms)
            # self.ttl_lattice.on()
            # self.ttl_lattice.off()
            self.lattice_upload()
            self.run_MOT(loading)  # 运行MOT，输入参数为装载时间,ms

            # self.ttl_guide.off()
            # self.ttl_guide.on()
            # self.ttl_lattice.on()
            # self.ttl_lattice.off()

            self.runMultiPGC()
            delay(self.lattDelay*ms)
            self.lattice_unload()
            # delay(self.RamanPulse * us)
            # self.ttl_lattice.off()
            # delay(20 * ms)
            # self.dds_cooling.sw.off()
            # delay(20 * us)

            # self.ttl_lattice.off()

            # self.ttl_guide.off()
            # self.ttl_magnet.off()
            # self.dds_cooling.sw.off()
            # self.ttl_repump.off()
            # delay(i*ms)


            self.detectFluPulse(hoding=hoding,t=t)
            # self.detectFluPulse(hoding=hoding,t=i)
            # self.detectFlu()
            # self.ttl_guide.off()
            # self.ttl_lattice.off()

            delay(10 * ms)


        if self.motAtLast:
            self.run_MOT(100)
            delay(1 * ms)

    