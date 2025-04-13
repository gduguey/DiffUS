# Understanding US Physics

## US Wave Formation

Let's begin by discussing the domain of sound waves. Audible waves range from 20 Hz to 20 kHz. Ultrasound (US) refers to sound waves with frequencies above 20 kHz. In medical imaging, clinical ultrasound typically uses frequencies between 2 MHz and 15 MHz — far above audible sound. These high-frequency, low-wavelength waves allow for finer spatial resolution. The relationship between wave speed, frequency, and wavelength is given by:

$$
c = f \lambda
$$

In ultrasound echography:

- A transducer (or probe) emits a pulse of US waves into the body.
- The waves reflect off internal structures with different acoustic properties.
- The time delay and amplitude of the returned echoes are used to estimate the depth and characteristics of tissues.

These waves propagate as pressure waves through the tissue. The equation above governs their behavior.

In biological tissues, we assume a typical speed of sound around $c = 1540 \, \text{m/s}$. To interpret echoes, we also need to understand the acoustic impedance of tissues. The acoustic impedance $Z$ characterizes how much resistance an ultrasound wave encounters as it passes through a medium and is defined as:

$$
Z = \rho c
$$

Where:
- $ \rho $ is the tissue density in $ \text{kg/m}^3 $,
- $ c $ is the speed of sound in that medium.

### Assumptions
For successful reconstruction of tissue structure, the following assumptions are made:

- (A) Sound travels in straight lines.
- (B) The speed of sound is uniform in all tissues.
- (C) A single pulse is emitted and received back at the transducer.
- (D) Attenuation of the wave is uniform throughout the medium.
- (E) Signals originate only from the main beam (not side lobes).

Most assumptions are reasonable. For example:
- (A), (C), and (E) follow from controlled probe design and beam focusing.
- (B) is supported by measurements showing that most soft tissues (excluding bone and air) have similar sound speeds, typically around 1540 m/s (Duck, 1990).

Given these assumptions, if a pulse is emitted at time $ t_1 $ and an echo is received at time $ t_2 $, then the estimated depth $ d $ of the reflecting interface is:

$$
d = \frac{c}{2} \times (t_2 - t_1)
$$

### Acoustic Impedance of Biological Tissues

By representing mediums as elements with acoustic impedance $z=p/v$ with $p$ local pressure waves and $v$ particle velocity, we can also write it as $z=\rho c$ where $\rho$ represents the density in $kg\space  m^{-3}$ and $c$ is the velocity in the medium.

The table below represents some values of bodily acoustic impedance

| Tissue | Value [Rayls]  | 
| ------ | ------- |
| Bone  | $6.46\times 10^6$  |  
| Blood  | $1.67\times 10^6$  |  
| Liver  | $1.66\times 10^6$  |  
| Kidney  | $1.64\times 10^6$  |  
| Fat  | $1.33\times 10^6$  |  
| Air  | $430$  |  

_Table of acoustic values (Hoskins, ?)_

Apart from bones, most bodily tissues seem to share similar values of Impedance around $1.6 \times 10^6 $.

![waves](img/waves.png)

To detect reflected waves, a change in medium (i.e., a discontinuity in impedance) is required. When a wave encounters such a boundary, the following conditions hold:

- The pressure and particle velocity must be continuous across the boundary.
- Assuming the same speed of sound across the boundary, the reflected pressure $ p_r $ relates to the incident pressure $p_i$ by:

$$
\frac{p_r}{p_i} = \frac{Z_2 - Z_1}{Z_1 + Z_2}
$$

- The reflected intensity ratio (reflection coefficient) is:

$$
\frac{I_r}{I_i} = \left( \frac{Z_2 - Z_1}{Z_1 + Z_2} \right)^2 = R_a^2
$$

Where:
- $ Z_1, Z_2 $ are the acoustic impedances of the first and second media,
- $ I_i $, $ I_r $, $ I_t $ are the incident, reflected, and transmitted wave intensities.
- $R_a$ is the reflection coefficient of the physical component.


### In practice: transducer implementation

Let's look again at a pulse. Upon arrival at a frontier, two coefficients $R_p$ and $T_p$, signalling reflection and transmission, are necessary to understand what is going on in detail. However, from the point of view of the transducer, the protocol is the following:

The transducer works as follows:
- Emits a short US pulse at a frequency \( f \),
- for a time of $T$, listen to returned echoes:
    - Note arrival of waves at $t_1,t_2,...t_k$ and their amplitude $a_1,a_2,...,a_k$
    - For each $t_i$, compute existing distance $d_i=\frac{c}{2}\times t_i$
    - Impute the value of the corresponding pixel by $a_i$ the amplitude of the received wave.

This yields a vector, corresponding to an arrau like $[0, 0, 0, a_1, 0, 0, 0, a_2, 0, a_3]$ corresponding to the different depths and corresponding amplitudes of the machine.


### B-Mode Imaging

B-mode (brightness mode) imaging is the most common ultrasound mode. It constructs 2D grayscale images by scanning multiple adjacent lines, where the brightness corresponds to echo amplitude.

Types of B-mode transducer formats:
- Linear
- Curvilinear
- Trapezoidal
- Radial

![Bmode_shapes](img/BmodeShapes.png)
_Fig. 1 B-Mode transducer shapes (Diagnostic Ultrasound: Physics and Equipment Fig. 14)_


### M-Mode Imaging

M-mode (motion mode) captures a single scan line over time. It is ideal for visualizing moving structures (e.g., heart valves). The vertical axis represents depth, and the horizontal axis represents time.

### Doppler Imaging

Doppler ultrasound is used to measure blood flow velocity using the Doppler effect. The frequency shift \( \Delta f \) between emitted and received signals reflects the relative motion of red blood cells:

$$
\Delta f \propto v \cos(\theta)
$$

Where \( v \) is the velocity of blood and \( \theta \) is the angle between the beam and the flow direction.

---

## MRI

will need to fill in 