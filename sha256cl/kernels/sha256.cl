// SHA 256 implementācija

#define SS0(x) (rotate(x, 7u) ^ rotate(x, 18u) ^ (x >> 3u))
#define SS1(x) (rotate(x, 17u) ^ rotate(x, 19u) ^ (x >> 10u))
#define S0(x) (rotate(x, 2u) ^ rotate(x, 13u) ^ rotate(x, 22u))
#define S1(x) (rotate(x, 6u) ^ rotate(x, 11u) ^ rotate(x, 25u))
#define CH(x, y, z) ((x & y) ^ (~x & z))
#define MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))

// pirmie 32 biti no kubsaknēm pirmajiem 64 pirmskaitļiem 2 - 311
__constant uint k[] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
					   0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
					   0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
					   0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
					   0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
					   0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
					   0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
					   0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// apstrādā vienu, konkrētu 512 bitu bloku
// 'input' satur apstrādājamo bitu bloku
// 'hash_output' ir 8 skaitļu masīvs, kas apstrādes beigās saturēs hash vērtību
// izvada 0, ja viss ok, -1, ja nav ok
int sha256(__constant uchar *input, uint length, uint *hash_output)
{
	// vienkāršības pēc apstrādāsim viena bloka ietvaros, tāpēc, ņemot vērā ziņojuma garumu un padding,
	// ziņojuma garums nedrīkst būt lielāks par 440 bitiem, lai viss ietilpstu vienā 512 bitu blokā
	// https://crypto.stackexchange.com/questions/54852/what-happens-if-a-sha-256-input-is-too-long-longer-than-512-bits
	if (length > (440 / 8))
	{
		return -1;
	}

	uint w[64];

	// pirmie 32 biti kv. saknei no pirmajiem 8 pirmskaitļiem 2 - 19 (no daļas aiz komata)
	hash_output[0] = 0x6a09e667;
	hash_output[1] = 0xbb67ae85;
	hash_output[2] = 0x3c6ef372;
	hash_output[3] = 0xa54ff53a;
	hash_output[4] = 0x510e527f;
	hash_output[5] = 0x9b05688c;
	hash_output[6] = 0x1f83d9ab;
	hash_output[7] = 0x5be0cd19;

	uchar chunk[64] = {0};

	// iekopē ievades tekstu
	for (int i = 0; i < length; i++)
	{
		chunk[i] = input[i];
	}

	// pēc prasībām ir jāpieliek '1' bits, pārējās baita vērtības attiecīgi ir nulles, atbilstoši SHA mainīgā 'K'
	// prasībām
	chunk[length] = 0b10000000;

	// padding galā jāpieliek ziņojuma garums kā 64 bitu big-endian skaitlis
	for (int i = 1; i <= 8; i++)
	{
		chunk[64 - i] = ((length * 8) >> ((i - 1) * 8)) & 0xFF;
	}

	// iekopē visus 512 bitus iekš w masīva (512/32 = 16 vērtības)
	// baiti jāieliek iekš 32 bitu vārdiem, lai pirmais baits būtu pirmais (skatoties no kreisās uz labo pusi),
	// tas jāpabīda pa kreisi pa 24, nākamie pa 16, 8, 0
	// attiecīgā solī nākamie 'mazāksvarīgie' biti ir nulles, tāpēc baitus šos baitus var konkatenēt ar OR (|) operatoru
	for (int i = 0; i < 16; i++)
	{
		w[i] = chunk[i * 4 + 0] << 24;
		w[i] |= chunk[i * 4 + 1] << 16;
		w[i] |= chunk[i * 4 + 2] << 8;
		w[i] |= chunk[i * 4 + 3];
	}

	// aizpilda pārējas 'w' vērtības
	for (int i = 16; i < 64; i++)
	{
		w[i] = w[i - 16] + SS0(w[i - 15]) + w[i - 7] + SS1(w[i - 2]);
	}

	uint a = hash_output[0];
	uint b = hash_output[1];
	uint c = hash_output[2];
	uint d = hash_output[3];
	uint e = hash_output[4];
	uint f = hash_output[5];
	uint g = hash_output[6];
	uint h = hash_output[7];

	for (int i = 0; i < 64; i++)
	{
		uint temp1 = h + S1(e) + CH(e, f, g) + k[i] + w[i];
		uint temp2 = S0(a) + MAJ(a, b, c);
		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}

	hash_output[0] += a;
	hash_output[1] += b;
	hash_output[2] += c;
	hash_output[3] += d;
	hash_output[4] += e;
	hash_output[5] += f;
	hash_output[6] += g;
	hash_output[7] += h;

	return 0;
}

size_t current_pw_size(__constant uint *offsets, uint password_count, uint char_count, uint idx)
{
	// not the last password
	if (idx < password_count - 1)
	{
		return offsets[idx + 1] - offsets[idx];
	}
	// last password, therefore we can't use the next offset
	else
	{
		return char_count - offsets[idx];
	}
}

__kernel void sha256_crack(__constant uchar *passwords, __constant uint *offsets,
						   uint password_count, uint char_count, __constant uint *target_hash,
						   __global atomic_int *cracked_idx)

{
	size_t idx = get_global_id(0);

	if (idx >= password_count)
	{
		return;
	}

	__constant uchar *my_password = passwords + offsets[idx];

	size_t pw_size = current_pw_size(offsets, password_count, char_count, idx);

	uint hash[8] = {0};

	int sha_result = sha256(my_password, pw_size, hash);

	bool match = true;

	for (int i = 0; i < 8; i++)
	{
		if (hash[i] != target_hash[i])
		{
			match = false;
			break;
		}
	}

	if (match)
	{
		atomic_store(cracked_idx, idx);
	}
}
