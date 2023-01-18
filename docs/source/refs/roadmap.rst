Development Roadmap
===================

Following is a loosely defined roadmap for the development of the codebase. The roadmap is subject to
change and is not a commitment to deliver specific features by specific dates or in the specified order.

Some of the features listed below are already implemented in the codebase, but are not yet documented
and/or tested. We will be working on improving the documentation and testing of these features in the
coming months.

**January 2023**

* |check_|  Experimental functional API
* Supported motion generators

  * |check_| Joint-space control
  * |check_| Differential inverse kinematics control
  * |uncheck| Riemannian Motion Policies (RMPs)

* Supported robots

  * |check_| Quardupeds: ANYmal-B, ANYmal-C, Unitree A1
  * |check_| Arms: Franka Emika Panda, UR10
  * |check_| Mobile manipulators: Franka Emika Panda and UR10 on Clearpath Ridgeback

* Supported sensors

  * |check_| Camera (non-parallelized)
  * |check_| Height scanner (non-parallelized)

* Included environments

  * |check_| classic: MuJoCo-style environments (ant, humanoid, cartpole)
  * |check_| locomotion: flat terrain for legged robots
  * |check_| rigid-object manipulation: end-effector tracking, object lifting

**February 2023**

* |uncheck| Add APIs for rough terrain generation
* |uncheck| Example on using the APIs in an Omniverse extension
* Supported motion generators

  * |uncheck| Operational-space control
  * |uncheck| Model predictive control (OCS2)

* Supported sensors

  * |uncheck| Height scanner (parallelized for terrains)

* Supported robots

  * |uncheck| Quardupeds: Unitree B1, Unitree Go1
  * |uncheck| Arms: Kinova Jaco2, Kinova Gen3, Sawyer, UR10e
  * |uncheck| Mobile manipulators: Fetch

* Included environments

  * |uncheck| locomotion: rough terrain for legged robots
  * |uncheck| rigid-object manipulation: in-hand manipulation, hockey puck pushing, peg-in-hole, stacking
  * |uncheck| deformable-object manipulation: cloth folding, cloth lifting

**March or April 2023**

* |uncheck| Add functional versions of all environments
* Supported sensors

  * |uncheck| Multi-camera support

* Included environments

  * |uncheck| deformable-object manipulation: fluid transfer, fluid pouring, soft object lifting

**May 2023**

* |uncheck| Stabilize APIs and release 1.0


.. |check| raw:: html

    <input checked=""  type="checkbox">

.. |check_| raw:: html

    <input checked=""  disabled="" type="checkbox">

.. |uncheck| raw:: html

    <input type="checkbox">

.. |uncheck_| raw:: html

    <input disabled="" type="checkbox">