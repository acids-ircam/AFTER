// inlets & outlets
inlets = 2
outlets = 2
// Global current model
var cur_prior = 0;
var cur_encoder = 0;
var cur_decoder = 0;
var timbre_recv = 0;
var timbre_unp = 0;
var cur_z = 0;
var cur_connect = [];
// Instance prefix
var prefix_z = "z";
var prefix_p = "";
// Object creation variable
var p = this.patcher

// Set prefix
function prefix(val_z, val_p)
{
	prefix_z = val_z
	prefix_p = val_p
}

// Create a given model
function create(model_name, prior) 
{
	if (model_name == "off")
	{
		_delete_current()
		return
	}
	else 
	{
		_delete_current()	
	}
	// Replace current model
	if (cur_z != 0)
		_delete_current()
	// Create encoder
	// var max_encoder = p.newdefault(100, 100, "nn~", model_name, "structure", 8192);
	// max_encoder.rect = [100, 100, 400, 25]
	// Create decoder
	var max_decoder = p.newdefault(100, 350, "nn~", model_name, "generate_timbre", 8192);
	max_decoder.rect = [100, 350, 850, 25]
	// Number of latents
	// var n_latents = max_encoder.getboxattr('numoutlets')
	// Gather all 
	cur_z = []
	// cur_encoder = max_encoder
	cur_decoder = max_decoder

	cur_connect.push(max_decoder)
	// cur_connect.push(max_encoder)

	// Make major in / out connections
	// p.connect(p.getnamed("model_props"), 0, max_encoder, 0)
	p.connect(p.getnamed("model_props"), 0, max_decoder, 0)
	p.connect(p.getnamed("plug_in"), 0, max_decoder, 0)
	p.connect(p.getnamed("msg_in"), 0, max_decoder, 0)
	p.connect(max_decoder, 0, p.getnamed("rave_out"), 0)
	cur_prior = 0
	// Create prior
	var main_plot = p.getnamed("main_plot");
	// var main_legend = p.getnamed("main_legend");

	main_plot.setattr("bkgndpict", model_name + ".png")
	// main_legend.setattr("pic", model_name + ".legend.png")
	// Latent connections
	// for (var z = 0; z < n_latents; z++)
	// {
		
	// 	var z_mix = p.newdefault(100 + (100 * z), 180, "mc.gen~", "rave");
	// 	z_mix.rect = [100 + (100 * z), 180, 185 + (100 * z), 180]
	// 	var z_mix_r = p.newdefault(110 + (100 * z), 140, "r", prefix_p+"_z");
	// 	z_mix_r.rect = [110 + (100 * z), 150, 150 + (100 * z), 150]
	// 	// p.connect(max_encoder, z, z_mix, 0)
	// 	p.connect(z_mix_r, 0, z_mix, 2)
	// 	if (prior)
	// 		p.connect(max_prior, z, z_mix, 1)
	// 	cur_connect.push(z_mix)
	// 	cur_connect.push(z_mix_r)
	// 	// Indirect connection
	// 	if (z < 4)
	// 	{
	// 		var z_send = p.newdefault(100 + (100 * z), 220 + (25 * z), "mc.send~", ""+prefix_z+""+z+"_e");
	// 		z_send.rect = [100 + (100 * z), 220 + (25 * z), 190 + (100 * z), 220 + (25 * z)]
	// 		p.connect(z_mix, 0, z_send, 0)
	// 		var z_recv = p.newdefault(100 + (100 * z), 225 + (25 * (z + 1)), "mc.receive~", ""+prefix_z+""+z);
	// 		z_recv.rect = [100 + (100 * z), 225 + (25 * (z + 1)), 190 + (100 * z), 225 + (25 * (z + 1))]
	// 		p.connect(z_recv, 0, max_decoder, z)
	// 		cur_connect.push(z_send)
	// 		cur_connect.push(z_recv)
	// 		cur_z.push(cur_connect)
	// 		continue
	// 	}
	// 	// Direct connection
	// 	p.connect(z_mix, 0, max_decoder, z)
	// 	cur_z.push(cur_connect)

		

	// }

	var timbre_recv = p.getnamed("timbre_receive")
	var timbre_unp = p.newdefault(100 + (100 * 8), 190, "mc.unpack~",8);
	// timbre_recv.rect = [100 + (100 * z), 180, 190 + (100 * z), 180]
	timbre_unp.rect = [100 + (100 * 8), 190, 190 + (100 * 8), 190]

	p.connect(timbre_recv, 0,timbre_unp,  0)

	for (var j = 0; j < 8; j++)
		{
			p.connect(timbre_unp, j,max_decoder,  1 + j)
		}

	
	// p.connect(timbre_recv, 1, n_latents-1, 0)
	// cur_connect.push(timbre_recv)
	cur_connect.push(timbre_unp)


	//### TIMBRE CONNECTION ###
	
	var max_encoder_timbre =  p.newdefault(2500, 1100, "nn~", model_name, "timbre", 8192);
	// max_encoder_timbre.rect = [2500, 1200, 400, 25]

	var receive_timbre = p.getnamed("receive_timbre");
	var pack_timbre = p.getnamed("pack_timbre");
	var enable_timbre = p.getnamed("enable_timbre");

	cur_connect.push(max_encoder_timbre)
	
	p.connect(receive_timbre, 0,max_encoder_timbre,  0);
	p.connect(enable_timbre, 0,max_encoder_timbre,  0);


	for (var j = 0; j < 8; j++)
		{
			p.connect(max_encoder_timbre, j ,pack_timbre,  j)
		}


	var sigm2l1 = p.getnamed("sigm2l1");
	var sigm2l2 = p.getnamed("sigm2l2");
	var packm2l = p.getnamed("packm2l");


	var unpackl2m = p.getnamed("unpackl2m");
	var snapl2m1= p.getnamed("snapl2m1");
	var snapl2m2 = p.getnamed("snapl2m2");

	var max_m2l =  p.newdefault(2900, 1100, "nn~", model_name, "map2latent");
	var max_l2m =  p.newdefault(3400, 1500, "nn~", model_name, "latent2map");


	p.connect(sigm2l1, 0 ,max_m2l,  0);
	p.connect(sigm2l2, 0 ,max_m2l,  1);

	for (var j = 0; j < 8; j++)
	{
		p.connect(max_m2l, j ,packm2l,  j)
	}
	
	for (var j = 0; j < 8; j++)
		{
			p.connect(unpackl2m, j ,max_l2m,  j)
		}
		
		p.connect(max_l2m, 0, snapl2m1, 0)
		p.connect(max_l2m, 1, snapl2m2, 0)

	cur_connect.push(max_m2l);
	cur_connect.push(max_l2m);

	outlet(1, "create")
	outlet(0, "bang")
}



// Create a given model
function encoder(model_name, prior) 
{
	if (cur_decoder == 0)
		return
	if (cur_prior)
		p.remove(cur_prior)
	p.remove(cur_encoder)
	// Create encoder
	var max_encoder = p.newdefault(100, 100, "mc.nn~", model_name, "encode");
	max_encoder.rect = [100, 100, 400, 25]
	// Number of latents
	var n_latents = max_encoder.getboxattr('numoutlets')
	// Decoder number of latents
	var d_latents = cur_decoder.getboxattr('numinlets')
	// Gather all 
	cur_encoder = max_encoder
	// Make major in / out connections
	p.connect(p.getnamed("model_props"), 0, max_encoder, 0)
	p.connect(p.getnamed("plug_in"), 0, max_encoder, 0)
	// Create prior
	cur_prior = 0
	if (prior)
	{
		var max_prior = p.newdefault(100, 100, "mc.nn~", model_name, "prior");
		max_prior.rect = [450, 100, 850, 25]
		cur_prior = max_prior
		p.connect(p.getnamed("temperature"), 0, max_prior, 0)
		p.connect(p.getnamed("model_props"), 0, max_prior, 0)
	}
	// Latent connections
	for (var z = 0; z < Math.min(n_latents, d_latents); z++)
	{
		cur_connect = cur_z[z]
		z_mix = cur_connect[0]
		z_mix_r = cur_connect[1]
		p.connect(max_encoder, z, z_mix, 0)
		if (prior)
			p.connect(max_prior, z, z_mix, 1)
	}
	outlet(1, "encoder")
}


// Create a given model
function decoder(model_name) 
{
	// Replace current model
	if (cur_encoder == 0)
		return
	p.remove(cur_decoder)
	// Create decoder
	var max_decoder = p.newdefault(100, 350, "mc.nn~", model_name, "decode");
	max_decoder.rect = [100, 350, 850, 25]
	// Decoder number of latents
	var d_latents = max_decoder.getboxattr('numinlets')
	// Number of latents
	var e_latents = cur_encoder.getboxattr('numoutlets')
	// Gather all 
	cur_decoder = max_decoder
	// Make major in / out connections
	p.connect(p.getnamed("model_props"), 0, max_decoder, 0)
	p.connect(max_decoder, 0, p.getnamed("rave_out"), 0)
	// Latent connections
	for (var z = 0; z < Math.min(d_latents, e_latents); z++)
	{
		cur_connect = cur_z[z]
		z_mix = cur_connect[0]
		// Indirect connection
		if (z < 4)
		{
			z_recv = cur_connect[3]
			p.connect(z_recv, 0, max_decoder, z)
			continue
		}
		// Direct connection
		p.connect(z_mix, 0, max_decoder, z)
	}
	outlet(1, "decoder")
}

function delete()
{
	if (cur_z != 0)
		_delete_current()
}

// delete current model
function _delete_current()
{	
	for (var item in cur_connect){
		p.remove(cur_connect[item])
	}


	var main_plot = p.getnamed("main_plot");
	main_plot.setattr("bkgndpict", "background_transparent.png")

	cur_connect = []

}
