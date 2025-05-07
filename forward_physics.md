# Modeling the forward process of wave propagation in the acoustic mediums

## Theoretical Basis

We have seen previously the physical basis for wave propagation between two medium, with a change in pressure waves at a boundary translating into variations in the impedances, and thus the apparition of non-zero reflection coefficients, and non-one transmission coefficients

Looking at the following diagram, we can be reminded of the changes at a boundary $i$. For our context, we will consider that each voxel boundary is a medium change. In the coming text, we will consider that a switch from a cell i to a following cell i+1 is a boundary i. This will be important in modeling the forward process.

Intuitively, there are exponentially many constraints on the transmission as every boundary change causes a change in reflected/transmitted change, and should thus influence all of the other reflections and transmissions further down the chain. A first simulation would thus have a ray propagating, reflecting, transmitting and a counter recording the "time of return": this is what actually happens in a transducer. Since we consider that speed is constant within the body, we can instead change the prism a bit to the following paradigm: by recording the amplitude of the wave in cell i, we can use that as the sound wave received forom the reflection at boundary i.

<!-- Check that this is enough and we don't need to do cumulative products of the waves,so that received wave at --> 

## Technical considerations

We now need to approach solving the differential equation obtained at boundary i. Assuming conservation, we have:
$$\texttt{Incoming waves:} g_i, d_{i+1}$$
$$\texttt{Outgoing waves:} g_{i+1} = R_{i,i+1}d_{i+1} + T_{i\rightarrow i+1}g_i, d_{i} = R_{i,i}g_{i} + T_{i+1 \rightarrow i} d_{i+1}$$

We can write this out as a full matrix, leading to a diagonal by block matrix that we can inverse. We will use, for $x$, $[g_0,d_0, ... g_n,d_n]$ and we will inverse $A^{-1}b$ with the edge conditions $b=[1, 0, ... 0, 0]$ to justify sending a wave on the left ($g_0=1$), and $d_{n+1}=0$. We can compute an example for two interface:

$$\begin{bmatrix}a & b \\ c & d\end{bmatrix}$$

Inversing us leads us to identify the $g$, $d$ corresponding to the left traveling and right traveling amplitudes of waves.

## One row examples

bla bla. 
