import numpy as np
import jax

# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from jax import device_put, devices
from jax import jit
from jax import config

config.update("jax_enable_x64", True)


class bispectrumExtractor:
    def __init__(self, L, Nmesh, kbinedges, verbose=True, low_mem=True, singleField=True) -> None:
        """Initializer of bispectrumExtractor
        Also calculates mesh of k-vectors

        Args:
            L (float): Boxsidelength
            Nmesh (int): Number of Gridcells along one dimension
            kbinedges (np.array): List of k-edges for bispectrum [h/Mpc]
            verbose (bool, optional): Whether to give extra-output to standard output. Defaults to True.
            low_mem (bool, optional): Whether algorithms with lower memory requirements should be used. Defaults to True.
        """
        self.L = L
        self.Nmesh = Nmesh
        if kbinedges != []:
            self.kbinedges = kbinedges
            self.Nks = len(kbinedges[0])
        self.prefactor = self.L**6 / self.Nmesh**9  # Prefactor for bispectrum
        # For n-point correlations, the prefactor needs to be L^(n-1)/N^(3*n)
        self.verbose = verbose
        self.low_mem = low_mem
        self.singleField=singleField

        if self.verbose:
            print("Finished setting members of bispectrumExtractor")
            print("Creating k-mesh")
        self.kmesh = self.createKmesh()

        self.calculateIk = jit(self.calculateIk)

    def createKmesh(self):
        """Creates mesh of k-vectors based on box sidelength and Nmesh

        Returns:
            jnp.array: Jax Numpy array, on device
        """
        idx, idy, idz = np.indices((self.Nmesh, self.Nmesh, self.Nmesh))
        idx = idx - idx.shape[0] / 2
        idy = idy - idy.shape[1] / 2
        idz = idz - idz.shape[2] / 2
        return jnp.sqrt(idx**2 + idy**2 + idz**2) * 2 * np.pi / self.L

    def applyMask(self, field, kmin, kmax):
        """Applies mask to field: If k values are outside of kmin or kmax field is set to zero

        Args:
            field (jnp.ndarray[complex]): 3d- fourier-transformed density field
            kmin (float): lower kbin edge
            kmax (float): upper kbin edge

        Returns:
            jnp.ndarray[complex]: filtered field
        """
        return field * ((self.kmesh <= kmax) & (self.kmesh >= kmin))

    def getFourierField(self, field_real):
        """Reads out real-space density field and gives back Fourier transformed field

        Args:
            field_real (np.ndarray): Density field in real space
        Returns:
            jnp.ndarray[complex]: Fourier transformed density field
        """

        dev_field_real = device_put(np.array(field_real, dtype=np.float32))
        field_fourier = jnp.fft.fftshift(jnp.fft.fftn(dev_field_real))
        del dev_field_real
        return field_fourier

    def calculateIk(self, field_fourier, kmin, kmax):
        """Calculates the Ik (inverse FFT of filtered field)

        Args:
            field_fourier (jnp.ndarray): Fourier transformed density field (not filtered)
            kmin (float): lower kbin edge
            kmax (float): upper kbin edge

        Returns:
            jnp.ndarray[float]: Ik value for this k bin
        """
        field_tmp = self.applyMask(field_fourier, kmin, kmax)

        field_tmp = jnp.fft.ifftn(jnp.fft.ifftshift(field_tmp)).real
        return field_tmp

    def calculateIks(self, field_fourier):
        """Calculates the Ik for all ks in kbinedges

        Args:
            field_fourier (jnp.ndarray): Fourier transformed density field (not filtered)

        Warning:
            The resulting Ik can be very large! Make sure this can fit in your RAM!

        Returns:
            np.ndarray: Array containing all Iks (on CPU)
        """
        Iks = np.zeros((self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=complex)
        for i in range(self.Nks):
            Iks[:, :, :, i] = self.calculateIk(
                field_fourier, self.kbinedges[0][i], self.kbinedges[1][i]
            )
        Iks = device_put(Iks, devices("cpu")[0])
        return Iks

    def calculateNorms(self):
        """Calculates the bispec normalization for all ks in kbinedges

        Warning:
            The resulting Norms-array can be very large! Make sure this can fit in your RAM!

        Returns:
            np.ndarray: Array containing all Norms (on CPU)
        """
        Ones = jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=float)
        Norms = np.zeros((self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=complex)
        for i in range(self.Nks):
            Norms[:, :, :, i] = self.calculateIk(
                Ones, self.kbinedges[0][i], self.kbinedges[1][i]
            )
        Norms = device_put(Norms, devices("cpu")[0])
        return Norms

    def calculateIk_Q(self):
        """Calculates the Ik integral over the qs for all ks in kbinedges (needed for calculating effective triangles)

        Warning:
            The resulting Ik can be very large! Make sure this can fit in your RAM!

        Returns:
            np.ndarray: Array containing all Iks (on CPU)
        """
        Iks = np.zeros(
            (self.Nmesh, self.Nmesh, self.Nmesh, self.Nks), dtype=np.complex64
        )
        for i in range(self.Nks):
            Iks[:, :, :, i] = self.calculateIk(
                self.kmesh, self.kbinedges[0][i], self.kbinedges[1][i]
            )
        Iks = device_put(Iks, devices("cpu")[0])
        return Iks

    def calculateBispectrumNormalization(
        self,
        mode="equilateral",
        custom_kbinedges_low=[],
        custom_kbinedges_high=[],
        precision=np.float64,
    ):
        """Calculates the Bispectrum Normalization. Only needs to be run once for all simulations with the same L and Nmesh.

            If `self.low_mem` is True, the I_ks are calculated on the fly which requires less memory but is slower for unequilateral triangles.
            Else, the I_ks are calculated once and stored in memory. This should usually be faster, but could run over the RAM.
            It could also trigger that CPU calculation instead of GPU calculation is performed, which massively decreases the speed.

        Args:
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral', 'custom', or 'all'. Defaults to 'equilateral'.
            custom_kbinedges_low (list, optional): lower edge of custom k-bins. Defaults to [].
            custom_kbinedges_high (list, optional): higher edge of custom k-bins. Defaults to [].
            precision (dtype, optional): float type of calculation. Defaults to np.float64
        Returns:
            list: normalizations for each triangle configuration.
        """
        normalization = []

        if self.low_mem:
            Ones = jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=precision)
        else:
            Norms = self.calculateNorms()

        if mode in {"equilateral", "all"}:
            for i in range(self.Nks):
                Norm1 = (self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i]) if self.low_mem else Norms[:, :, :, i])

                if mode == "equilateral":
                    normalization.append(jnp.sum(Norm1**3))
                    continue

                j_range = range(i, self.Nks) if self.singleField else range(self.Nks)

                for j in j_range:
                    if i==j:
                        Norm2=Norm1
                    else:
                        Norm2=(self.calculateIk(Ones, self.kbinedges[0][j], self.kbinedges[1][j]) if self.low_mem else Norms[:, :, :, j])

                    k_range = range(j, self.Nks) if self.singleField else range(self.Nks)

                    for k in k_range:
                        if (self.kbinedges[2][k] > self.kbinedges[2][i] + self.kbinedges[2][j]):
                            continue

                        if k==j:
                            Norm3=Norm2
                        elif k==i:
                            Norm3=Norm1
                        else:
                            Norm3=(self.calculateIk(Ones, self.kbinedges[0][k], self.kbinedges[1][k]) if self.low_mem else Norms[:, :, :, k])

                        normalization.append(jnp.sum(Norm1 * Norm2 * Norm3))

        elif mode == "custom":
            if custom_kbinedges_low is None or custom_kbinedges_high is None or  len(custom_kbinedges_low)==0 or len(custom_kbinedges_high)==0:
                raise ValueError(
                    f"custom_kbinedges need to be provided if mode is {mode}"
                )
            if not self.low_mem and self.verbose:
                print(
                    "Warning: Using low-memory mode with custom bin edges; high-memory optimization not applicable."
                )
            for low, high in zip(custom_kbinedges_low, custom_kbinedges_high):
                Norm=self.calculateIk(Ones, low[0], high[0])
                Norm*=self.calculateIk(Ones, low[1], high[1])
                Norm*=self.calculateIk(Ones, low[2], high[2])
                
                normalization.append(jnp.sum(Norm))
                del Norm

        else:
            raise ValueError(
                f"Mode cannot be {mode}, must be either 'all', 'equilateral', or 'custom'."
            )

        return normalization

    def calculateEffectiveTriangle(
        self,
        mode="equilateral",
        custom_kbinedges_low=[],
        custom_kbinedges_high=[],
        precision=np.float64,
    ):
        """Calculates the Effective Triangles. Only needs to be run once for all simulations with the same L and Nmesh.

        If `self.low_mem` is True, the I_ks are calculated on the fly which requires less memory but is slower for unequilateral triangles.
        Else, the I_ks are calculated once and stored in memory. This should usually be faster, but could run over the RAM.
        It could also trigger that CPU calculation instead of GPU calculation is performed, which massively decreases the speed.

        Args:
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral', 'custom', or 'all'. Defaults to 'equilateral'.
            custom_kbinedges_low (list, optional): lower edge of custom k-bins. Defaults to [].
            custom_kbinedges_high (list, optional): higher edge of custom k-bins. Defaults to [].
            precision (dtype, optional): float type of calculation. Defaults to np.float64

        Returns:
            list: Effective triangle configurations.
        """

        effectiveKs = []
        Ones = (None if not self.low_mem
            else jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=precision))
        Norms = None if self.low_mem else self.calculateNorms()
        Ik_Qs = None if self.low_mem else self.calculateIk_Q()

        if mode in {"equilateral", "all"}:
            for i in range(self.Nks):
                Norm1 = (self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i]) if self.low_mem else Norms[:, :, :, i])
                Ik_Q1 = (self.calculateIk(self.kmesh, self.kbinedges[0][i], self.kbinedges[1][i]) if self.low_mem else Ik_Qs[:, :, :, i])

                if mode == "equilateral":
                    k = jnp.sum(Norm1**2 * Ik_Q1)
                    effectiveKs.append([k, k, k])
                    continue

                j_range = range(i, self.Nks) if self.singleField else range(self.Nks)


                for j in j_range:
                    if i==j:
                        Norm2=Norm1
                        Ik_Q2=Ik_Q1
                    else:
                        Norm2=(self.calculateIk(Ones, self.kbinedges[0][j], self.kbinedges[1][j]) if self.low_mem else Norms[:, :, :, j])
                        Ik_Q2 = (self.calculateIk(self.kmesh, self.kbinedges[0][j], self.kbinedges[1][j]) if self.low_mem else Ik_Qs[:, :, :, j])
                    
                    k_range = range(j, self.Nks) if self.singleField else range(self.Nks)

                    for k in k_range:
                        if (self.kbinedges[2][k] > self.kbinedges[2][i] + self.kbinedges[2][j]):
                            continue

                        if k==j:
                            Norm3=Norm2
                            Ik_Q3=Ik_Q2
                        elif k==i:
                            Norm3=Norm1
                            Ik_Q3=Ik_Q1
                        else:
                            Norm3=(self.calculateIk(Ones, self.kbinedges[0][k], self.kbinedges[1][k]) if self.low_mem else Norms[:, :, :, k])
                            Ik_Q3=(self.calculateIk(self.kmesh, self.kbinedges[0][k], self.kbinedges[1][k]) if self.low_mem else Ik_Qs[:, :, :, k])


                        effectiveKs.append([
                                jnp.sum(Ik_Q1 * Norm2 * Norm3),
                                jnp.sum(Norm1 * Ik_Q2 * Norm3),
                                jnp.sum(Norm1 * Norm2 * Ik_Q3)])

        elif mode == "custom":
            if custom_kbinedges_low is None or custom_kbinedges_high is None:
                raise ValueError(
                    f"custom_kbinedges need to be provided if mode is {mode}"
                )
            if not self.low_mem and self.verbose:
                print(
                    "Warning: Using low-memory mode with custom bin edges; high-memory optimization not applicable."
                )
            for low, high in zip(custom_kbinedges_low, custom_kbinedges_high):
                Norm2, Norm3 = [self.calculateIk(Ones, low[i], high[i]) for i in (1, 2)]
                Ik_Q1 = self.calculateIk(self.kmesh, low[0], high[0])
                k1 = jnp.sum(Ik_Q1 * Norm2 * Norm3)

                Norm1 = self.calculateIk(Ones, low[0], high[0])
                Ik_Q2 = self.calculateIk(self.kmesh, low[1], high[1])
                k2 = jnp.sum(Norm1 * Ik_Q2 * Norm3)

                Ik_Q3 = self.calculateIk(self.kmesh, low[2], high[2])
                k3 = jnp.sum(Norm1 * Norm2 * Ik_Q3)

                effectiveKs.append([k1, k2, k3])

        else:
            raise ValueError(
                f"Mode cannot be {mode}, must be 'all', 'equilateral', or 'custom'."
            )

        return effectiveKs

    def calculateBispectrum(self,field_real,mode="equilateral",custom_kbinedges_low=[],custom_kbinedges_high=[],field_real2=None,field_real3=None):
        """Calculates the unnormalized Bispectrum using either the low-memory or high-memory code.

        If `self.low_mem` is True, the I_ks are calculated on the fly which requires less memory but is slower for unequilateral triangles.
        Else, the I_ks are calculated once and stored in memory. This should usually be faster, but could run over the RAM.
        It could also trigger that CPU calculation instead of GPU calculation is performed, which massively decreases the speed.


        Args:
            field_real (np.ndarray): Real space density field (in numpy binary format).
            mode (str, optional): Which k-triangles to consider. Can be 'equilateral', 'custom', or 'all'. Defaults to 'equilateral'.
            custom_kbinedges_low (list, optional): lower edge of custom k-bins. Defaults to [].
            custom_kbinedges_high (list, optional): higher edge of custom k-bins. Defaults to [].
            precision (dtype, optional): float type of calculation. Defaults to np.float64

        Returns:
            list: Unnormalized bispectrum for each triangle configuration.
        """

        if ((field_real2 is None) or (field_real3 is None)) and not self.singleField:
            raise ValueError("You have set multi-field mode (singleField=False) but have only provided one density field!")
        
        if self.singleField and not ((field_real2 is None) and (field_real3 is None)):
            raise ValueError("You have set single-field mode (singleField=True) but have provided multiple density fields!")

        fields_fourier=[]
        fields_fourier.append(self.getFourierField(field_real))

        if self.singleField:
            if self.verbose:
                print("Doing Fourier Transformation of density field")
        else:
            if self.verbose:
                print("Doing Fourier Transformation of density fields")
            fields_fourier.append(self.getFourierField(field_real2))
            fields_fourier.append(self.getFourierField(field_real3))

        if self.verbose:
            print("Doing Bispec calculation")

        bispec=[]

        Iks=[]
        for f in fields_fourier:
            Iks.append(None if (self.low_mem or mode=="custom") else self.calculateIks(f))


        if mode in {"equilateral", "all"}:
            for i in range(self.Nks):
                Ik1 = (self.calculateIk(fields_fourier[0], self.kbinedges[0][i], self.kbinedges[1][i]) if self.low_mem 
                                        else Iks[0][:, :, :, i])
                if mode == "equilateral":
                    if self.singleField:
                        Ik2=Ik1
                        Ik3=Ik1
                    else:
                        Ik2 = (self.calculateIk(fields_fourier[1], self.kbinedges[0][i], self.kbinedges[1][i]) if self.low_mem 
                                        else Iks[1][:, :, :, i])
                        Ik3 = (self.calculateIk(fields_fourier[2], self.kbinedges[0][i], self.kbinedges[1][i]) if self.low_mem 
                                        else Iks[2][:, :, :, i])
                    bispec.append(jnp.sum(Ik1*Ik2*Ik3))
                    del Ik1, Ik2, Ik3
                    continue

                j_range = range(i, self.Nks) if self.singleField else range(self.Nks)

                for j in j_range:
                    if self.singleField:
                        if i==j:
                            Ik2=Ik1
                        else:
                            Ik2=(self.calculateIk(fields_fourier[0], self.kbinedges[0][j], self.kbinedges[1][j]) if self.low_mem 
                                 else Iks[0][:, :, :, j])
                    else:
                        Ik2=(self.calculateIk(fields_fourier[1], self.kbinedges[0][j], self.kbinedges[1][j]) if self.low_mem 
                                 else Iks[1][:, :, :, j])

                    k_range = range(j, self.Nks) if self.singleField else range(self.Nks)

                    for k in k_range:
                        if (self.kbinedges[2][k] > self.kbinedges[2][i] + self.kbinedges[2][j]):
                            continue
                        if self.singleField:
                            if k==j:
                                Ik3=Ik2
                            elif k==i:
                                Ik3=Ik1
                            else:
                                Ik3=(self.calculateIk(fields_fourier[0], self.kbinedges[0][k], self.kbinedges[1][k]) if self.low_mem 
                                 else Iks[0][:, :, :, k])
                        else:
                            Ik3=(self.calculateIk(fields_fourier[2], self.kbinedges[0][k], self.kbinedges[1][k]) if self.low_mem 
                                 else Iks[2][:, :, :, k])
                       
                        result = float(jnp.sum(Ik1 * Ik2 * Ik3))
                        bispec.append(result)

                        del Ik3
                    if not (self.singleField and i==j):
                        del Ik2
                del Ik1
        elif mode=="custom":
            if custom_kbinedges_low is None or custom_kbinedges_high is None:
                raise ValueError(f"custom_kbinedges need to be provided if mode is {mode}")
            

            if not self.low_mem and self.verbose:
                print("Warning: Using low-memory mode with custom bin edges; high-memory optimization not applicable.")

            for i in range(len(custom_kbinedges_high)):
                Ik=self.calculateIk(fields_fourier[0], custom_kbinedges_low[i][0], custom_kbinedges_high[i][0])
                if self.singleField:
                    Ik*=self.calculateIk(fields_fourier[0], custom_kbinedges_low[i][1], custom_kbinedges_high[i][1])
                else:
                    Ik*=self.calculateIk(fields_fourier[1], custom_kbinedges_low[i][1], custom_kbinedges_high[i][1])
                
                if self.singleField:
                    Ik*=self.calculateIk(fields_fourier[0], custom_kbinedges_low[i][2], custom_kbinedges_high[i][2])
                else:
                    Ik*=self.calculateIk(fields_fourier[2], custom_kbinedges_low[i][2], custom_kbinedges_high[i][2])

                bispec.append(jnp.sum(Ik))
                del Ik


        else:
            raise ValueError(f"Mode cannot be {mode}, must be either 'all', 'equilateral', or 'custom'.")

        return bispec
    


    def calculatePowerspectrum(self, field_real):
        """Calculates the unnormalized Powerspectrum

        Args:
            field_real (np.ndarray): Real space density field (in numpy binary format)



        Returns:
            list: unnormalized bispectrum for each triangle configuration
        """

        if self.verbose:
            print("Doing Fourier Transformation of density field")

        field_fourier = self.getFourierField(field_real)

        if self.verbose:
            print("Doing Powerspec calculation")
        powerspec = []

        for i in range(self.Nks):
            Ik = np.real(
                self.calculateIk(
                    field_fourier, self.kbinedges[0][i], self.kbinedges[1][i]
                )
            )
            tmp = jnp.sum(Ik**2)
            powerspec.append(tmp)

        return powerspec


    def calculatePowerspectrumNormalization(self, precision=np.float64):

        Ones = jnp.ones((self.Nmesh, self.Nmesh, self.Nmesh), dtype=precision)

        if self.verbose:
            print("Doing Powerspec normalization calculation")

        normalization = []

        for i in range(self.Nks):
            Norm = np.real(
                self.calculateIk(Ones, self.kbinedges[0][i], self.kbinedges[1][i])
            )
            tmp = jnp.sum(Norm**2)
            normalization.append(tmp)

        return normalization
