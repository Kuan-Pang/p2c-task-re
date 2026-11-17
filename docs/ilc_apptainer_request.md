# Draft request to the ILC admins: Apptainer on `il-cpu`

*Draft for you to send — not sent. Edit freely. Context: PLAN.md §9.*

---

**Subject:** Request: root Apptainer package (with its AppArmor profile) on the il-cpu partition

Hi ILC admins,

I run benchmark-evaluation jobs with the [Harbor](https://docs.harborframework.com) framework. On a cluster without
Docker, Harbor's only local option is its Apptainer/Singularity backend. That backend runs every task with
`apptainer exec --fakeroot --containall …`, using a prebuilt `.sif` image. Nothing is built on the node, and no
setuid or root access is needed at run time.

It cannot run on ILC today:

- `which apptainer singularity` finds nothing on ampere2 (Ubuntu 24.04.4, kernel 6.8.0-134).
- `kernel.apparmor_restrict_unprivileged_userns = 1`, so unprivileged user namespaces are blocked:
  `unshare --user --map-root-user id` fails with `write failed /proc/self/uid_map: Operation not permitted`.
  `--fakeroot` needs them.
- The Apptainer documentation says that on Ubuntu 24.04 user namespaces are available only to packages
  "installed as root and set up with an apparmor configuration file", so a user-space install does not help.

Could you install the **Apptainer package from the official PPA or .deb, as root, together with the AppArmor
profile it ships** (`/etc/apparmor.d/apptainer`) on the il-cpu nodes? That leaves the system-wide userns
restriction in place and allows user namespaces for Apptainer only. If a subuid/subgid range is needed for
`--fakeroot`, a range for my user would also be needed.

The jobs are CPU-only and network-light. Each container makes outbound HTTPS calls to model APIs only.
I would test on one node first, if you prefer to start there.

Thanks,
<name>

---

**Before sending:** only ampere2 was probed. Running `which apptainer; sysctl kernel.apparmor_restrict_unprivileged_userns`
on one il-cpu node (e.g. inside `srun -p il-cpu --pty bash`) would confirm the same state there.
