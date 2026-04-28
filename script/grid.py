import torch
import math
import torch.nn.functional as F


class Grid:
    def __init__(
        self,
        pos:           torch.Tensor,
        chg:           torch.Tensor,
        eps:           torch.Tensor | None = None,
        sig:           torch.Tensor | None = None,
        log_eps:       torch.Tensor | None = None,
        log_sig:       torch.Tensor | None = None,
        grid_interval: float = 0.3,
        grid_buffer:   float = 5.0,
        center:        torch.Tensor | None = None,
        grid_size:     torch.Tensor | None = None,
        grid_coords:   torch.Tensor | None = None,
        taus:          dict[str, list[float]] | None = None,
    ):
        self.pos       = pos
        self.chg       = chg
        self.eps       = eps
        self.sig       = sig
        self.log_eps   = log_eps
        self.log_sig   = log_sig

        self.dtype  = pos.dtype
        self.device = pos.device

        self.grid_interval = grid_interval
        self.grid_buffer   = grid_buffer
        self.taus          = taus

        self.center      = pos.mean(dim=0) if center is None else center
        self.grid_size   = self.set_grid_size() if grid_size is None else grid_size
        self.grid_coords = self.build_grid_coords() if grid_coords is None else grid_coords


    # ── Grid Setup ─────────────────────────────────────────────────────────────

    def set_grid_size(self):
        """Calculate bounding box size with buffer around atomic positions."""
        max_vals = self.pos.max(dim=0).values
        min_vals = self.pos.min(dim=0).values
        lengths  = max_vals - min_vals
        size     = [math.ceil(s) + 2.0 * self.grid_buffer for s in lengths.tolist()]
        return torch.tensor(size, device=self.device, dtype=self.dtype)


    def build_grid_coords(self):
        """
        Generate 3D FCC grid coordinates.
        Nearest-neighbor distance is maintained at grid_interval.
        """
        center = self.center
        half   = 0.5 * self.grid_size
        d      = self.grid_interval
        device = self.device
        dtype  = self.dtype

        # 2D hexagonal basis vectors
        a1 = torch.tensor([d, 0.0], device=device, dtype=dtype)
        a2 = torch.tensor([0.5 * d, 0.5 * math.sqrt(3.0) * d], device=device, dtype=dtype)

        x_min, x_max = center[0] - half[0], center[0] + half[0]
        y_min, y_max = center[1] - half[1], center[1] + half[1]

        nx = int(math.ceil(float(half[0].item()) / d)) + 3
        ny = int(math.ceil(float(half[1].item()) / float(a2[1].item()))) + 3

        ii = torch.arange(-nx, nx + 1, device=device)
        jj = torch.arange(-ny, ny + 1, device=device)
        I, J = torch.meshgrid(ii, jj, indexing="ij")

        IJ  = torch.stack([I.reshape(-1), J.reshape(-1)], dim=1).to(dtype)
        xy0 = IJ[:, 0:1] * a1[None, :] + IJ[:, 1:2] * a2[None, :]

        # Keep only points inside the bounding box
        in_box = (
            (xy0[:, 0] + center[0] >= x_min) & (xy0[:, 0] + center[0] <= x_max) &
            (xy0[:, 1] + center[1] >= y_min) & (xy0[:, 1] + center[1] <= y_max)
        )
        xy0 = xy0[in_box]

        # Z layers with FCC stacking phases
        dz    = math.sqrt(2.0 / 3.0) * d
        z_min, z_max = center[2] - half[2], center[2] + half[2]
        nz    = int(math.ceil(float(half[2].item()) / dz)) + 2
        kk    = torch.arange(-nz, nz + 1, device=device, dtype=dtype)
        z_levels = center[2] + kk * dz
        keep_z   = (z_levels >= z_min) & (z_levels <= z_max)
        z_levels = z_levels[keep_z]
        kk_int   = kk[keep_z].to(torch.int64)

        # Apply FCC layer shifts (0, s, 2s) cyclically
        s      = (a1 + a2) / 3.0
        phase  = torch.remainder(kk_int, 3)
        shifts = torch.zeros((z_levels.numel(), 2), device=device, dtype=dtype)
        shifts[phase == 1] = s
        shifts[phase == 2] = 2.0 * s

        coords_list = []
        for li in range(z_levels.numel()):
            xy = xy0 + shifts[li][None, :]
            z  = torch.full((xy.size(0), 1), z_levels[li], device=device, dtype=dtype)
            coords_list.append(torch.cat([xy + center[None, :2], z], dim=1))

        return torch.cat(coords_list, dim=0)


    # ── Attributes ─────────────────────────────────────────────────────────────

    def softplus_attr(self, add=0.001):
        """Convert log-space parameters to physical values via softplus."""
        self.eps = F.softplus(self.log_eps) + add
        self.sig = F.softplus(self.log_sig) + add


    # ── Grid Fields ────────────────────────────────────────────────────────────

    def get_grid(self, store=True):
        """
        Compute molecular field grids for all field types and tau values.
        - density : Gaussian density field
        - charge  : Coulomb-like potential (total, positive, negative)
        - epsilon : LJ-like VDW potential (repulsion + attraction)
        If store=True, saves to self.G. Otherwise returns the dict.
        """
        pos         = self.pos.view(1, 1, 1, -1, 3)
        chg         = self.chg.view(1, 1, 1, -1)
        sig         = self.sig.view(1, 1, 1, -1)
        eps         = self.eps.view(1, 1, 1, -1)
        grid_coords = self.grid_coords.unsqueeze(-2)

        sq_r = torch.sum((grid_coords - pos) ** 2, dim=-1)
        r    = torch.sqrt(sq_r)

        grids = {}
        for gtype, tau_list in self.taus.items():
            for tau in tau_list:

                if gtype == 'density':
                    w     = tau * sig
                    g_den = torch.exp(-sq_r / (2 * w ** 2)).sum(dim=-1)
                    grids[f'{gtype}_{tau}'] = [gtype, tau, g_den]

                elif gtype == 'charge':
                    denom = torch.sqrt(sq_r + tau ** 2 + 1e-12)
                    grids[f'{gtype}_{tau}']     = [gtype, tau, torch.sum(chg / denom, dim=-1)]
                    grids[f'{gtype}_pos_{tau}'] = [gtype, tau, torch.sum(chg.clamp(min=0.0) / denom, dim=-1)]
                    grids[f'{gtype}_neg_{tau}'] = [gtype, tau, torch.sum(chg.clamp(max=0.0) / denom, dim=-1)]

                elif gtype == 'epsilon':
                    w     = tau * sig
                    r_min = (2.0 ** (1.0 / 6.0)) * sig
                    g_rep = (4.0 * eps * torch.exp(-0.5 * (r / w) ** 2)).sum(dim=-1)
                    g_att = (4.0 * eps * torch.exp(-(r - r_min) ** 2 / (2.0 * w ** 2))).sum(dim=-1)
                    grids[f'{gtype}_rep_{tau}'] = [gtype, tau, g_rep]
                    grids[f'{gtype}_att_{tau}'] = [gtype, tau, g_att]

        if store:
            self.G = grids
        else:
            return grids


    # ── Atomic Importance ──────────────────────────────────────────────────────

    def get_atomic_ranks(self):
        """
        Rank atoms by their contribution to all field types.
        Used to select the most representative COMPRESS sites from the AA molecule.
        Returns indices sorted from least to most important.
        """
        pos         = self.pos.view(1, 1, 1, -1, 3)
        chg         = self.chg.view(1, 1, 1, -1)
        sig         = self.sig.view(1, 1, 1, -1)
        eps         = self.eps.view(1, 1, 1, -1)
        grid_coords = self.grid_coords.view(-1, 1, 3).unsqueeze(0)

        sq_r = torch.sum((grid_coords - pos) ** 2, dim=-1)
        r    = torch.sqrt(sq_r)
        tau  = 1.0

        # Density contribution
        C_den = torch.exp(-sq_r / (2.0 * (tau * sig) ** 2)).sum(dim=2).squeeze(0).squeeze(0)

        # Charge contribution
        C_chg = (chg / torch.sqrt(sq_r + tau ** 2)).abs().sum(dim=2).squeeze(0).squeeze(0)

        # VDW contribution (repulsion + attraction)
        r_min = (2.0 ** (1.0 / 6.0)) * sig
        C_rep = (4.0 * eps * torch.exp(-0.5 * (r / (tau * sig)) ** 2)).sum(dim=2).squeeze(0).squeeze(0)
        C_att = (4.0 * eps * torch.exp(-(r - r_min) ** 2 / (2.0 * (tau * sig) ** 2))).sum(dim=2).squeeze(0).squeeze(0)

        # Normalize each contribution and sum
        def norm(x): return x / x.sum().clamp_min(1e-12)
        C_total = norm(C_den) + norm(C_chg) + norm(C_rep) + norm(C_att)

        return torch.argsort(C_total, descending=False)
